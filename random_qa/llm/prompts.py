from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate, PromptType
from typing import ClassVar

from random_qa.utils import sanitize_multiline_string


class BaseTask:
    """Base class to define task-specific prompt templates"""
    system_template: ClassVar[str | None] = None
    task_template: ClassVar[str | None] = None
    response_format: ClassVar[dict | None] = None

    @classmethod
    def as_chat_kwargs(cls, prompt_variables: dict, separate_system_prompt: bool = True) -> dict:
        """Format the prompt templates by filling in the placeholders.

        Parameters:
            data: key-value pairs for substituting placeholders in the prompt templates
            separate_system_prompt: If `False`, system and task prompt are combined as a single chat message. Otherwise,
                there is chat message for the system prompt and for the task prompt, respectively.

        Returns: list of chat message objects
        """
        assert cls.system_template is not None
        assert cls.task_template is not None
        system_prompt = PromptTemplate(sanitize_multiline_string(cls.system_template)).format(**prompt_variables)
        task_prompt = PromptTemplate(sanitize_multiline_string(cls.task_template)).format(**prompt_variables)

        chat_kwargs = {}
        if separate_system_prompt:
            chat_kwargs["messages"] = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=task_prompt)
            ]
        else:
            chat_kwargs["messages"] = [ChatMessage(role=MessageRole.USER, content=system_prompt + "\n\n" + task_prompt)]

        if cls.response_format is not None:
            chat_kwargs["response_format"] = cls.response_format

        return chat_kwargs
    
    @classmethod
    def as_completion_kwargs(cls, prompt_variables: dict) -> dict:
        completion_kwargs = cls.as_chat_kwargs(prompt_variables, separate_system_prompt=False)
        completion_kwargs["prompt"] = completion_kwargs.pop("messages")[0].content
        return completion_kwargs


class QueryGenerationTask(BaseTask):
    system_template = "You are a Neo4j graph database expert that helps translating user questions into Cypher statements."
    task_template = """
    Task: Generate Cypher statement to query a knowledge graph.
    Instructions:
    The Cypher statement must be inside a Markdown codeblock delimited by three consecutive backticks.
    Use only the provided labels, relationship types and properties from the schema.
    Always use meaningful aliases in the RETURN clause.

    Schema:
    {schema}

    Note:
    Do not include any apologies in your responses. Do not generate alternative statements.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Before generating the Cypher query, explain which path patterns, functions, labels, relationships and properties must be taken into account.
    
    The question is: {question}

    Let's think step-by-step.
    """


class QueryCorrectionTask(BaseTask):
    system_template = "You are a Neo4j graph database expert that helps finding and fixing errors in Cypher statements."
    task_template = """
    Task: Fix the errors in a Cypher statement.
    Instructions:
    The Cypher statement must be inside a Markdown codeblock delimited by three consecutive backticks.
    Use only the provided labels, relationship types and properties from the schema.

    Schema:
    {schema}

    Note: Do not include any apologies in your responses. Do not generate alternative statements.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Before generating the fixed Cypher statement, reason how the errors can be fixed.

    The question is: {question}
    The faulty Cypher statement is:
    ```
    {query}
    ```
    The error messages are:
    {errors}
    
    Let's think step-by-step.
    """


class AnswerGenerationTask(BaseTask):
    system_template = "You are an assistant that gives helpful and human understandable answers."
    task_template = """
    Task: Answer a question using the result of a Cypher query.
    Instructions:
    Keep the answer as short as possible. 
    Use only the information from the provided schema, query and query result to give the answer.

    Schema:
    {schema}

    Query:
    ```
    {query}
    ```

    Query result:
    ```json
    {result}
    ```

    Note: Do not include any explainations in your responses. Do not mention that you based the result on the given information.
    The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
    If the provided information is not enough to answer the question, say that it is not possible to give an answer.

    The question is: {question}

    The answer is:
    """

class ClosedBookAnswerGenerationTask(BaseTask):
    system_template = "You are an assistant that gives helpful and human understandable answers."
    task_template = """
    Task: Answer a question using the result of a Cypher query.
    Instructions:
    Keep the answer as short as possible. 
    Use only the information from the provided schema, query and query result to give the answer.

    Schema:
    {schema}

    Note: Do not include any explainations in your responses. Do not mention that you based the result on the given information.
    Unfortunately, the Cypher query cannot be executed at the moment. Instead, try to guess the correct answer using the information provided in the schema or based on your internal knowledge.

    The question is: {question}

    The answer or best possible guess is:
    """


class CorrectnessGradingTask(BaseTask):
    # Reference: https://github.com/quotient-ai/judges/blob/main/judges/graders/correctness.py
    # Reference: https://arxiv.org/pdf/2306.05685
    # Reference: https://github.com/langchain-ai/openevals/blob/main/python/openevals/prompts/correctness.py
    system_template = "You are a fair judge assistant tasked with providing clear, objective assessment based on specific criteria."
    task_template = """
    Task: Assess the factual correctness of a generated answer by comparing it to the given ground truth answer.
    Instructions:
    Write a brief assessment that judges the quality of the generated answer based on the score rubrics.
    After writing the justification, give a score that is an integer between 1 (factually incorrect) and 5 (factually correct).

    Question: {question}

    Generated answer: {generated_answer}

    Ground truth answer: {ground_truth_answer}

    Score rubrics:
    Score 1: The generated answer does not match with the ground truth answer at all.
    Score 2: The generated answer lacks most of the facts from the ground truth answer.
    Score 3: The generated answer contains some facts from the ground truth answer but is missing information or contains incorrect claims.
    Score 4: The generated answer covers most of the facts from the ground truth answer but can be improved slightly.
    Score 5: The generated answer matches the ground truth answer exactly.

    Note: Please do not generate any other opening, closing, and explanations.
    Refer to the provided score rubrics to evaluate the generated answer.
    The output must be a JSON object containing the justification as a string and the score as an integer between 1 and 5.
    Do not set the score to 0 since the lowest possible score is 1.

    Assessment:
    """
    # Reference: https://platform.openai.com/docs/guides/structured-outputs?format=without-parse#how-to-use
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "assessment",
            "schema": {
                "type": "object",
                "properties": {
                    "justification": {"type": "string"},
                    "score": {"type": "integer"}
                },
                "required": ["justification", "score"],
                "additionalProperties": False
            },
            "strict": True
        }
    }