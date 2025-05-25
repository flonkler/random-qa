import asyncio
import logging
import os
import sys

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType, PydanticUndefined

from argparse import ArgumentParser
from typing_inspect import is_literal_type, get_args

from random_qa.config import EvaluationConfig, InferenceConfig, SetupDatabaseConfig, CreateDatasetConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Dict[str, FieldInfo]
commands = {
    "inference": {
        "description": "Run inference and store the results in a file",
        "args": {
            "results_path": Field(description="", required=True),
            "samples_path": Field(description="", required=True),
            **InferenceConfig.model_fields,
        }        
    },
    "evaluation": {
        "description": "Evaluate the inference results and store the metrics in a file",
        "args": {
            "metrics_path": Field(description="Lorem", required=True),
            "samples_path": Field(description="", required=True),
            "results_path": Field(description="", required=True),
            **EvaluationConfig.model_fields,
        }        
    },
    "generate-samples": {
        "description": "Generate the QA samples and store them in a file",
        "args": {
            "samples_path": Field(description="Lorem", required=True),
            **CreateDatasetConfig.model_fields,
        }        
    },
    "setup-database": {
        "description": "Generate the KG and write it to the Neo4j database",
        "args": SetupDatabaseConfig.model_fields,
    },
    "schema-description": {
        "description": "Print the schema description"
    },
}


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m random_qa",
        description="Command-line interface for the random_qa module",
    )
    command_subparser = parser.add_subparsers(metavar="command", dest="command")
    for command, command_info in commands.items():
        p = command_subparser.add_parser(command, help=command_info["description"])
        if "args" not in command_info:
            continue

        command_args: dict[str, FieldInfo] = command_info["args"]
        for name, field_info in command_args.items():
            if field_info.annotation is not None:
                field_type = field_info.annotation
            elif field_info.default is not PydanticUndefined and field_info.default is not None:
                field_type = type(field_info.default)
            else:
                field_type = str
            
            if is_literal_type(field_type):
                field_type = type(get_args(field_type, evaluate=True)[0])
            
            if name.endswith("dir") or name.endswith("path"):
                metavar = "PATH"
            elif name.endswith("uri"):
                metavar = "URI"
            elif name.endswith("llm"):
                metavar = "LLM"
            else:
                metavar = field_type.__name__.upper()
            # TODO: Use field info to obtain help text, required, etc.
            p.add_argument(
                f"--{name.replace('_', '-')}", metavar=metavar, dest=name, required=field_info.is_required(),
                type=field_type, help=field_info.description
            )

    return parser


async def main():
    parser = get_argparser()
    args = parser.parse_args()

    # Remove unset arguments from the dictionary
    args_dict = {k: v for k, v in args.__dict__.items() if v is not None}
    if args.command == "inference":
        import pandas as pd
        from random_qa.cli import run_inference, open_json, init_env
        init_env()
        df_samples = pd.DataFrame(open_json(args.samples_path, mode="r")).set_index("sample_id")
        try:
            df_results_old = pd.DataFrame(open_json(args.results_path, mode="r")).set_index("sample_id")
            df_samples = df_samples[~df_samples.index.isin(df_results_old.index)]
            logger.info("Loaded existing results file %s", os.path.abspath(args.results_path))
            if len(df_samples) == 0:
                logger.info("Results file is already complete, skipping inference")
                return
        except Exception:
            df_results_old = None
        
        df_results_new = await run_inference(df_samples, InferenceConfig.from_env(**args_dict))
        if df_results_old is None:
            df_results = df_results_new
        else:
            df_results = pd.concat((df_results_new, df_results_old)).sort_index()
        
        open_json(args.results_path, mode="w", data=df_results.reset_index(drop=False).to_dict(orient="records"))
    elif args.command == "evaluation":
        import pandas as pd
        from random_qa.cli import run_evaluation, open_json, init_env
        init_env()
        df_samples = pd.DataFrame(open_json(args.samples_path, mode="r")).set_index("sample_id")
        df_results = pd.DataFrame(open_json(args.results_path, mode="r")).set_index("sample_id")
        df_metrics_old = None
        try:
            df_metrics_old = pd.DataFrame(open_json(args.metrics_path, mode="r")).set_index("sample_id")
            df_results = df_results[~df_results.index.isin(df_metrics_old.index)]
            logger.info("Loaded existing metrics file %s", os.path.abspath(args.metrics_path))
            if len(df_results) == 0:
                logger.info("Metrics file is already complete, skipping evaluation")
                return
        except Exception:
            df_metrics_old = None

        df_metrics_new = await run_evaluation(df_samples, df_results, EvaluationConfig.from_env(**args_dict))
        if df_metrics_old is None:
            df_metrics = df_metrics_new
        else:
            df_metrics = pd.concat((df_metrics_old, df_metrics_new)).sort_index()
        
        open_json(args.metrics_path, mode="w", data=df_metrics.reset_index(drop=False).to_dict(orient="records"))
    elif args.command == "generate-samples":
        import pandas as pd
        from random_qa.cli import create_dataset, open_json, init_env
        init_env()
        df_samples = create_dataset(CreateDatasetConfig.from_env(**args_dict))
        open_json(args.samples_path, mode="w", data=df_samples.reset_index(drop=False).to_dict(orient="records"))
    elif args.command == "setup-database":
        from random_qa.cli import setup_database, init_env
        init_env()
        setup_database(SetupDatabaseConfig.from_env(**args_dict))
    elif args.command == "schema-description":
        from random_qa.graph import kg_schema
        print(kg_schema.describe())
    else:
        logger.critical("Unknown command %r", args.command)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
