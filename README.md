# RANDOM-QA

> [!NOTE]  
> Link to the published thesis will be added after the review of the supervisors.

This repository contains the source code, datasets and results of my master's thesis _"RANDOM-QA: Question Answering mit Large Language Models und Knowledge Graphs für die Radio-Access-Network-Domäne"_.

## Repository structure

| Folder/File | Usage |
| ----------- | ----- |
| 📁 `data/` | Datasets used to generate the Knowledge Graph and the evaluation dataset |
| 📁 `experiments/` | Results of the conducted experiments (metrics, outputs, etc.) |
| 📁 `notebooks/` | Jupyter notebooks to create visualization, analyze results and interact with the implementation |
| 📁 `random_qa/` | Implementation of the KGQA system and helper functions |
| 📄 `.env` | General environment variables (e.g., API keys) for the project |
| 📄 `docker-compose.yaml` | Configuration of the Neo4j Docker container |
| 📄 `requirements.txt` | List of all Python dependencies |

## Setup

The implementation relies on several Python packages that must be installed. It is recommended to install them in a virtual environment and to use Python version 3.10+.

```bash
pip install -r requirements.txt
```

Some functions require access to a Neo4j database (e.g., to store the KG or to run queries). To setup a local Neo4j instance, a Docker container can be started using `docker-compose`.

``` bash
docker-compose up -d
```

## Usage

The implementation provides a command-line interface to interact with the KGQA system and to conduct the experiments. The following segments will explain the available commands and their options. Since these steps depend on each other, it is important to run them in order.

Alternatively, the methods and classes of the `random_qa` module can also be accessed directly. An example is shown in a [Jupyter notebook](notebooks/workflow.ipynb).

### Step 1: Setup database

In the first step the KG will be generated using the datasets in `data/` directory. By default the local Neo4j instance will be used to store the KG.

```bash
python -m random_qa setup-database --data-dir="data/"
```

After restarting the Neo4j container, the data is persisted in a volume. Also, the setup command will take existing nodes and relationships into account. Hence, running the setup multiple times will not result in any duplicates. In case the Neo4j container should be resetted to an empty database, removing the container is not sufficient. Instead the volume must be removed as well.

```bash
# Remove the associated volumes after stopping the container
docker-compose down --volumes
```

### Step 2: Generate samples

The second step is to create the evaluation dataset. This file is the foundation for all of experiments. The samples will be generated based on predefined QA templates.

```bash
python -m random_qa generate-samples \
    --templates-path="data/qa_templates.yaml" \
    --samples-path="experiments/samples_rep=5_seed=123.json.gz" \
    --seed=123 \
    --repetition=5
```

**Options:**

- `--templates-path`: Path to the YAML file that defines all QA templates
- `--samples-path`: Path of the evaluation dataset (i.e., the output)
- `--seed`: Seed for the random number generator to ensure reproducible outputs
- `--repetition`: Specifies how often a single template can used to generate samples


### Step 3: Run inference

After the evaluation dataset has been created, the samples can be processed by the KGQA system. The system configuration can be varied with regards to the LLMs and the QA scenario. The generated outputs (i.e., queries and answers) will be written to a file such that it can be inspected and evaluated later. 

```bash
python -m random_qa inference \
    --samples-path="experiments/samples_rep=5_seed=123.json.gz" \
    --results-path="experiments/inference_open-book_llm=gpt-4o.json.gz" \
    --scenario="open-book" \
    --query-llm="gpt-4o" \
    --answer-llm="gpt-4o"
```

**Options:**

- `--samples-path`: Path of the evaluation dataset
- `--results-path`: Path of the inference results (i.e., the output)
- `--scenario`: QA scenario which can be either "open-book", "closed-book" or "oracle"
- `--query-llm`/`--answer-llm`: Name of the LLM that is used to generate the queries and answers, respectively

### Step 4: Evaluate inference results

In the last step, the inference results can be evaluated using multiple metrics.

```bash
python -m random_qa evaluation \
    --samples-path="experiments/samples_rep=5_seed=123.json.gz" \
    --results-path="experiments/inference_open-book_llm=gpt-4o.json.gz" \
    --metrics-path="experiments/metrics_open-book_llm=gpt-4o.json.gz" \
    --eval-llm="gpt-4o"
```

**Options:**

- `--samples-path`: Path of the evaluation dataset which is used as the ground truth
- `--results-path`: Path of the inference results
- `--metrics-path`: Path of the evaluation results (i.e., the output)
- `--eval-llm`: Name of the LLM that is used to assess the answer correctness

## Customization

In this study, five LLMs were used in the experiments which were either self-hosted or accessed via a custom API. Because of that, the presets may not applicable to other environments. Instead, there are two possible ways to support other LLMs as well.

1. The LLM name can be prefixed with `openai:` to indicate that the standard [OpenAI implementation of llama-index](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai/) should be used. For example, `openai:gpt-4o-mini` refers to GPT-4o-mini provided by OpenAI. The required API key must be set via the environment variable `OPENAI_API_KEY`.
1. To use custom LLMs or other providers the implementation in [`random_qa/llm/models.py`](random_qa/llm/models.py) must be adjusted.
