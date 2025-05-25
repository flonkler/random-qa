import asyncio
import itertools
import neo4j
import os
import json
import logging
import gzip
import sys
import yaml

from datetime import datetime
from dotenv import load_dotenv
from nltk import download as nltk_download
from pandas import DataFrame
from pydantic import BaseModel
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm as tqdm_fallback
from typing import Literal

from random_qa.workflows import InferenceWorkflow, EvaluationWorkflow
from random_qa.config import InferenceConfig, EvaluationConfig, CreateDatasetConfig, SetupDatabaseConfig
from random_qa.llm import get_model_by_name
from random_qa.eval import BERTScorerWrapper, generate_samples, load_templates
from random_qa.graph import kg_schema, QueryValidator
from random_qa.graph.loader import (
    load_regions_dataframe,
    load_sites_dataframe,
    load_services_dataframe,
    load_manufacturers_dataframe,
    load_cells_dataframe,
    load_mobile_antennas_dataframe,
    load_microwave_antennas_dataframe,
    load_operators_dataframe,
    load_tiles_dataframe,
    load_buildings_dataframe,
    load_pois_dataframe,
)
from random_qa.graph.generator import (
    generate_regions,
    generate_sites,
    generate_services,
    generate_manufacturers,
    generate_cells,
    generate_operators,
    generate_tiles,
    generate_antennas,
    generate_pois
)
from random_qa.graph.heuristics import (
    compute_cell_coverage,
    compute_site_connection
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use standard tqdm as default
tqdm = tqdm_fallback

def _is_ipython_env() -> bool:
    """Detect if current environment is an IPython runtime"""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _setup_notebook_tqdm() -> None:
    """Replace global `tqdm` interface with notebook widget"""
    from IPython.display import display_html
    # Adjust CSS styling of widget to better match with dark themes
    display_html(
        """<style>
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground);
            --jp-widgets-font-size: var(--vscode-editor-font-size);
        }  
        </style>
        <div>Loaded <code>tqdm</code> widget for notebooks</div>""",
        raw=True
    )
    # Replace global tqdm class
    global tqdm
    tqdm = tqdm_notebook


def init_env() -> None:
    # TODO: Setup logging
    if _is_ipython_env():
        # Load env file from the project's root directory
        root_dir = os.path.abspath("..")
        load_dotenv(os.path.join(root_dir, ".env"))
        # Modify global `tqdm` interface
        _setup_notebook_tqdm()
    else:
        load_dotenv()

    # Load additional dependencies
    nltk_download("stopwords", quiet=True)
    nltk_download("punkt", quiet=True)


def open_json(path: str, mode: Literal["r", "w"], data: dict | None = None) -> dict:
    """Open a JSON file for reading or writing data. 
    
    Args:
        path: Path to the JSON file. If it ends with ".gz", gzip will be used for compression or decompression.
        mode: Determines whether to write (w) or read (r) the file
        data: Object that should be written to the file

    Returns:
        Data that was read from the file or written to the file.
    """
    if path.endswith(".gz"):
        f = gzip.open(path, mode=mode + "t", encoding="utf-8")
    else:
        f = open(path, mode=mode, encoding="utf-8")

    def _encoder_fallback(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        else:
            raise TypeError(f"Type {type(obj)} is not JSON serializable")

    with f:
        if mode == "r":
            data = json.load(f)
        else:
            json.dump(data, f, default=_encoder_fallback, ensure_ascii=False)

    return data


def create_dataset(config: CreateDatasetConfig) -> DataFrame:
    """Generate a dataset of samples
    
    Args:
        config: Object specifying the execution configurations (see `CreateDatasetConfig`)

    Returns:
        Dataframe with QA samples
    """
    logger.info("Generating QA samples with seed=%d and repetition=%d", config.seed, config.repetition)
    driver = neo4j.GraphDatabase.driver(config.neo4j_uri)
    with driver.session() as session:
        samples = map(
            lambda x: x.model_dump(mode="json"),
            generate_samples(session, load_templates(config.templates_path), repeat=config.repetition, seed=config.seed)
        )
        df = DataFrame(tqdm(samples, desc="Generating samples", unit=" samples")).set_index("sample_id")

    logger.info("Generated %d samples", len(df))
    return df


def setup_database(config: SetupDatabaseConfig) -> None:
    """Generate the KG and write it to Neo4j
    
    Args:
        config: Object specifying the execution configurations (see `SetupDatabaseConfig`)
    """
    # Load input dataframes
    logger.info("Loading dataframes")
    df_services = load_services_dataframe()
    df_manufacturers = load_manufacturers_dataframe()
    df_operators = load_operators_dataframe()
    gdf_sites = load_sites_dataframe(config.data_dir)
    gdf_sites = compute_site_connection(gdf_sites)
    gdf_pois = load_pois_dataframe(config.data_dir)
    gdf_mobile_antennas = load_mobile_antennas_dataframe(config.data_dir, gdf_sites)
    gdf_microwave_antennas = load_microwave_antennas_dataframe(gdf_sites, df_manufacturers)
    gdf_regions = load_regions_dataframe(config.data_dir)
    gdf_buildings = load_buildings_dataframe(config.data_dir)
    gdf_tiles = load_tiles_dataframe(gdf_regions)
    gdf_cells = load_cells_dataframe(config.data_dir, gdf_mobile_antennas, df_services)
    gdf_cells = compute_cell_coverage(gdf_cells, gdf_tiles, gdf_buildings, df_services)
    # Setup database connection
    driver = neo4j.GraphDatabase.driver(config.neo4j_uri)
    with driver.session() as session:
        # Create index for each label
        for i, cls in enumerate(kg_schema.nodes):
            label = cls.__name__
            session.run(f"CREATE INDEX {label.lower()}_id_index IF NOT EXISTS FOR (n:{label}) ON n.id")
        logger.info("Successfully created %d indices", i + 1)
        # Generate nodes and relationships and write them to the database
        iterator = itertools.chain(
            generate_regions(gdf_regions),
            generate_operators(df_operators),
            generate_services(df_services),
            generate_tiles(gdf_tiles),
            generate_manufacturers(df_manufacturers),
            generate_pois(gdf_pois, gdf_regions, gdf_tiles),
            generate_sites(gdf_sites, gdf_regions, gdf_pois),
            generate_antennas(gdf_mobile_antennas, gdf_microwave_antennas),
            generate_cells(gdf_cells),
        )
        for i, obj in tqdm(enumerate(iterator), desc="Writing nodes and relationships", unit="obj"):
            obj.write_to_db(session)
        logger.info("Successfully created %d nodes and relationships", i + 1)



async def run_inference(df_samples: DataFrame, config: InferenceConfig) -> DataFrame:
    """Perform inference on the samples dataset

    Args:
        df_samples: Dataframe with QA samples
        config: Object specifying the execution configurations (see `InferenceConfig`)

    Returns:
        Dataframe with inference results (i. e., generated queries, generated answerd and task details)
    """
    if config.samples_offset != "":
        logger.warning("Only samples starting from %s will be considered", config.samples_offset)
        df_samples = df_samples[df_samples.sample_id >= config.samples_offset]
    
    if config.samples_limit > 0:
        logger.warning("Samples are limited to %d", config.samples_limit)
        df_samples = df_samples.sample(config.samples_limit)

    if len(df_samples) == 0:
        raise ValueError("Samples dataframe is empty!")

    workflow = InferenceWorkflow(
        verbose=False, num_concurrent_runs=config.workflow_concurrent_runs, timeout=config.workflow_timeout
    )
    run_kwargs = {
        "query_llm": get_model_by_name(config.query_llm, config.default_llm_params),
        "answer_llm": get_model_by_name(config.answer_llm, config.default_llm_params),
        "schema": kg_schema.describe(),
        "query_validator": QueryValidator(schema=kg_schema),
        "neo4j_driver": neo4j.AsyncGraphDatabase.driver(config.neo4j_uri),
        "scenario": config.scenario,
    }
    handlers = [
        workflow.run(
            sample_id=sample_id,
            question=row.question,
            ground_truth_query=row.query,
            ground_truth_query_result=row.query_result,
            **run_kwargs
        ) for sample_id, row in df_samples.iterrows()
    ]
    results = []
    success, failed = 0, 0
    logger.info("Running inference on %d samples", len(df_samples))
    for result in tqdm(asyncio.as_completed(handlers), total=len(df_samples)):
        try:
            results.append(await result)
            success += 1
        except Exception:
            logger.exception("Inference workflow failed")
            failed += 1
    logger.info("Finished inference of %d runs successfully (%d runs failed)", success, failed)
    return DataFrame(results).set_index("sample_id")


async def run_evaluation(df_samples: DataFrame, df_results: DataFrame, config: EvaluationConfig) -> DataFrame:
    """Evaluate the inference results using the samples dataset

    Args:
        df_samples: Dataframe with QA samples
        df_results: Dataframe with inference results
        config: Object specifying the execution configurations (see `EvaluationConfig`)
    
    Returns:
        Dataframe with evaluation results (i. e., scores of metrics)
    """
    workflow = EvaluationWorkflow(
        verbose=False, num_concurrent_runs=config.workflow_concurrent_runs, timeout=config.workflow_timeout
    )
    with BERTScorerWrapper(model_name=config.bert_model_name, device=config.bert_model_device) as bert_scorer:
        run_kwargs = dict(
            evaluator_llm=get_model_by_name(config.eval_llm, model_params=config.default_llm_params),
            bert_scorer=bert_scorer,
        )
        handlers = [
            workflow.run(
                sample_id=sample_id,
                question=df_samples.loc[sample_id].question,                
                generated_query=result.query,
                generated_query_result=result.query_result,
                generated_answer=result.answer,
                ground_truth_query=df_samples.loc[sample_id].query,
                ground_truth_query_result=df_samples.loc[sample_id].query_result,
                ground_truth_answer=df_samples.loc[sample_id].answer,
                **run_kwargs
            )
            for sample_id, result in df_results.iterrows()
        ]
        logger.info("Running evaluation on %d samples", len(df_results))
        results = []
        success, failed = 0, 0
        for result in tqdm(asyncio.as_completed(handlers), total=len(df_results)):
            try:
                results.append(await result)
                success += 1
            except Exception:
                logger.exception("Evaluation workflow failed")
                failed += 1

    logger.info("Finished evaluation of %d runs successfully (%d runs failed)", success, failed)
    return DataFrame(results).set_index("sample_id")
