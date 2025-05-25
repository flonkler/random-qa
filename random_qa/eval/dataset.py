import itertools
import jinja2
import neo4j
import math
import random
import re
import yaml
from pandas import DataFrame

from enum import Enum
from pydantic import BaseModel
from typing import Iterable, Iterator, Any


class LabelEnum(Enum):
    ZERO_HOP = "0-hop"
    ONE_HOP = "1-hop"
    TWO_HOP = "2-hop"
    MULTI_HOP = "multi-hop"
    AGGREGATION = "aggregation"
    CONSTRAINT = "constraint"
    ALGORITHM = "algorithm"


class QATemplate(BaseModel):
    template_id: int
    candidates: str | None
    questions: list[str]
    query: str
    answer: str
    labels: list[LabelEnum]


class QASample(BaseModel):
    sample_id: str
    question: str
    query: str
    query_result: list[dict]
    answer: str
    labels: list[LabelEnum]


def _human_readable_list(value: Iterable[str]) -> str:
    result = ""
    array = list(value)
    if len(array) == 1:
        return array[0]
    
    for i, element in enumerate(array):
        if i == len(array) - 1:
            result += " and "
        elif i != 0:
            result += ", "
        result += element
    return result


def _map_format(value: Iterable[Any], format: str) -> Iterator[str]:
    for element in value:
        yield format % element


def _reduce_whitespaces(value: str) -> str:
    """Remove unnecessary whitespace characters a string (e.g., a rendered Jinja template)"""
    return re.sub(r"\s+", " ", value.replace("\n", " ")).strip()


def _cycling_product(left: list[Any], right: list[Any]) -> Iterator[tuple[Any, Any]]:
    """Cycle through two sequences and output non-repeating pairs. Works similar to `itertools.product` but yields pairs
    in a differnt order. At most `len(left) * len(right)` pairs can generated without repetition.
    
    Parameters:
        left: List of elements for the left-hand side of the pairs.
        right: List of elements for the right-hand side of the pairs.

    Yields:
        Tuple `(a, b)` where `a` is an element from the left list and `b` from the right.
    """
    wrap_after = math.lcm(len(left), len(right))
    for i in range(len(left) * len(right)):
        j = i // wrap_after
        yield left[i % len(left)], right[(i + j) % len(right)]


def load_templates(path: str) -> list[QATemplate]:
    with open(path, mode="r") as f:
        templates = list(QATemplate(**template) for template in yaml.safe_load(f))
    return templates


def generate_samples(
    session: neo4j.Session, templates: list[QATemplate], repeat: int = 5, seed: int = 123
) -> Iterator[QASample]:
    """Randomly generate evaluation samples based on templates.

    Args:
        session: Neo4j session to run querys (e.g., to obtain the ground truth answer).
        templates: List of question templates.
        repeat: Specifies how many sample should be generated for each template.
        seed: Seed for random number generator to ensure reproducible outputs.
    
    Yields:
        Data objects that store the input (i.e., question), ground truth output (i.e., query and answer) as well as 
        an indentifier and the labels of the sample.
    """
    jinja_env = jinja2.Environment()
    jinja_env.filters["human_readable_list"] = _human_readable_list
    jinja_env.filters["map_format"] = _map_format
    # Generate multiple samples from a single template
    for template in templates:
        if template.candidates is not None:
            # Execute candidates query
            candidates = session.run(template.candidates).data()
            if len(candidates) == 0:
                raise ValueError(f"Candidates query of template {template.template_id} produces an empty result!")
        else:
            candidates = [None]

        if len(candidates) > repeat:
            # Choose a random selection of elements that is deterministic for a fixed template ID and seed
            candidates = random.Random(template.template_id * seed).sample(candidates, k=repeat)

        # Iterate of a randomized sequence of candidates and question templates
        for i, (candidate, question_template) in zip(range(repeat), _cycling_product(candidates, template.questions)):
            # Render Jinja templates based on the chosen candidate
            question = jinja_env.from_string(question_template).render(candidate=candidate)
            query = jinja_env.from_string(template.query).render(candidate=candidate)
            # Execute ground truth query and use result to determine the ground truth answer
            result = session.run(query).data()
            if len(result) == 0:
                raise ValueError(f"Ground truth query of template {template.template_id} produces an empty result!")
            answer = jinja_env.from_string(template.answer).render(result=result, candidate=candidate)
            if answer.strip() == "":
                raise ValueError(f"Ground truth answer of template {template.template_id} is empty!")
            yield QASample(
                sample_id=f"{template.template_id}_{i + 1}",
                question=_reduce_whitespaces(question),
                query=query,
                query_result=result,
                answer=_reduce_whitespaces(answer),
                labels=template.labels
            )
