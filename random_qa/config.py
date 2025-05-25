import os

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_BERT_MODEL_NAME = "microsoft/deberta-xlarge-mnli"
DEFAULT_BERT_MODEL_DEVICE = "cuda"


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_env(cls, **data):
        for field in cls.model_fields.keys():
            value = os.getenv(field.upper())
            if value is not None:
                data[field] = value
        return cls(**data)


class CreateDatasetConfig(BaseConfig):
    templates_path: str = Field(required=True)
    seed: int = Field(required=False, default=123)
    repetition: int = Field(required=False, default=5)
    neo4j_uri: str = Field(required=False, default=DEFAULT_NEO4J_URI)


class SetupDatabaseConfig(BaseConfig):
    data_dir: str = Field(required=True)
    neo4j_uri: str = DEFAULT_NEO4J_URI


class ExperimentConfig(BaseConfig):
    default_llm_max_tokens: int = 1024
    default_llm_temperature: float = 0.0
    default_llm_seed: int = 123

    workflow_concurrent_runs: int = 1
    workflow_timeout: int = 500
    
    @property
    def default_llm_params(self) -> dict:
        return {
            "max_tokens": self.default_llm_max_tokens,
            "temperature": self.default_llm_temperature,
            "seed": self.default_llm_seed,
        }


class InferenceConfig(ExperimentConfig):
    scenario: Literal["open-book", "closed-book", "oracle"]
    query_llm: str
    answer_llm: str
    neo4j_uri: str = DEFAULT_NEO4J_URI
    samples_offset: str = ""
    samples_limit: int = 0


class EvaluationConfig(ExperimentConfig):
    eval_llm: str
    bert_model_name: str = DEFAULT_BERT_MODEL_NAME
    bert_model_device: str = DEFAULT_BERT_MODEL_DEVICE
