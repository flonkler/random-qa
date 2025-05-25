import neo4j
import time

from llama_index.core.workflow import Event, StopEvent, Context, step
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from neo4j.exceptions import CypherSyntaxError, CypherTypeError, DatabaseError

from random_qa.llm import CustomLLM, QueryGenerationTask, QueryCorrectionTask, AnswerGenerationTask, ClosedBookAnswerGenerationTask
from random_qa.workflows.base import BaseWorkflow, LLMResponseEvent, TrackedEvent, SetupDoneEvent
from random_qa.graph import QueryValidator
from random_qa.utils import extract_codeblock


class RetriesExceededEvent(Event):
    pass

class QueryGeneratedEvent(LLMResponseEvent, TrackedEvent):
    pass

class AnswerGeneratedEvent(LLMResponseEvent, TrackedEvent):
    pass

class QueryExecutedEvent(TrackedEvent):
    query: str
    result: list[dict]
    duration: float

class QueryValidatedEvent(TrackedEvent):
    query: str
    has_changes: bool

class QueryResultValidatedEvent(Event):
    query: str
    result: list[dict]

class QueryFailedEvent(TrackedEvent):
    query: str | None = None
    reason: str
    errors: list[str]
    success: bool = False

class OpenBookScenarioEvent(Event):
    pass

class ClosedBookScenarioEvent(Event):
    pass

class OracleScenarioEvent(Event):
    query: str
    result: list[dict] 


class InferenceWorkflow(BaseWorkflow):
    max_result_size: int = 50
    max_attempts: int = 3

    @step
    async def decide_scenario(
        self, ctx: Context, ev: SetupDoneEvent
    ) -> OpenBookScenarioEvent | ClosedBookScenarioEvent | OracleScenarioEvent:
        scenario = await ctx.get("scenario")
        if scenario == "open-book":
            return OpenBookScenarioEvent()
        elif scenario == "closed-book":
            return ClosedBookScenarioEvent()
        elif scenario == "oracle":
            return OracleScenarioEvent(
                query=await ctx.get("ground_truth_query"),
                result=await ctx.get("ground_truth_query_result")
            )

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=3, delay=5))
    async def generate_query(
        self, ctx: Context, ev: OpenBookScenarioEvent | QueryFailedEvent
    ) -> QueryGeneratedEvent | RetriesExceededEvent:
        """Generate LLM response that contains a query"""
        attempt = 1 if isinstance(ev, OpenBookScenarioEvent) else ev.attempt + 1
        if attempt > self.max_attempts:
            # Abort workflow if the maximum number of attempts is reached
            return RetriesExceededEvent()
    
        prompt_variables = {
            "question": await ctx.get("question"),
            "schema": await ctx.get("schema")
        }
        if isinstance(ev, QueryFailedEvent):
            # Construct query correction prompt if the previous attempt has failed
            chat_kwargs = QueryCorrectionTask.as_chat_kwargs({
                **prompt_variables,
                "query": ev.query or "",
                "errors": "\n".join(f"- {error}" for error in ev.errors)
            })
        else:
            # Construct query generation prompt if it is the first attempt
            chat_kwargs = QueryGenerationTask.as_chat_kwargs(prompt_variables)
        
        llm: CustomLLM = await ctx.get("query_llm")
        response = await llm.achat(**chat_kwargs)
        return QueryGeneratedEvent(
            task_name="generate_query",
            attempt=attempt,
            raw=response.message.content,
            **response.additional_kwargs
        )

    @step
    async def validate_query(self, ctx: Context, ev: QueryGeneratedEvent) -> QueryValidatedEvent | QueryFailedEvent:
        """Extract query from LLM response and check schema compliance"""
        query = extract_codeblock(ev.raw)
        if query is None or query == "":
            return QueryFailedEvent(
                task_name="generate_query",
                attempt=ev.attempt,
                reason="format_error",
                errors=["The generated response does not contain a valid Markdown codeblock."],
            )

        # Check schema compliance
        validator: QueryValidator = await ctx.get("query_validator")
        validator.validate(query)
        # Use automatically repaired version instead of the original query
        query = validator.fixed_query

        if validator.has_errors:
            return QueryFailedEvent(
                task_name="generate_query",
                attempt=ev.attempt,
                reason="schema_error",
                errors=validator.errors,
                query=query,
            )
        else:
            return QueryValidatedEvent(
                task_name="generate_query",
                attempt=ev.attempt,
                query=query,
                has_changes=validator.has_changes,
            )

    @step
    async def execute_query(self, ctx: Context, ev: QueryValidatedEvent) -> QueryExecutedEvent | QueryFailedEvent:
        """Execute the generated Cypher query"""
        driver: neo4j.AsyncDriver = await ctx.get("neo4j_driver")
        try:
            session_start = time.perf_counter()
            result = await self._run_query_session(driver, ev.query)
            session_duration = time.perf_counter() - session_start
            return QueryExecutedEvent(
                task_name="execute_query",
                attempt=ev.attempt,
                query=ev.query,
                result=result,
                duration=session_duration
            )
        except (CypherTypeError, CypherSyntaxError, DatabaseError) as ex:
            return QueryFailedEvent(
                task_name="execute_query",
                attempt=ev.attempt,
                query=ev.query,
                reason="syntax_error",
                errors=[ex.message]  # Use error message from the exception
            )
        
    @step
    async def validate_result(self, ctx: Context, ev: QueryExecutedEvent) -> QueryResultValidatedEvent | QueryFailedEvent:
        """Perform plausibility checks on the query result"""
        if len(ev.result) > self.max_result_size:
            return QueryFailedEvent(
                task_name="execute_query",
                attempt=ev.attempt,
                query=ev.query,
                reason="result_too_big",
                errors=[
                    f"The query result has too many entries ({len(ev.result)}, but only {self.max_result_size} or "
                    "fewer are allowed). The size can be reduced by using aggregations or the DISTINCT keyword."
                ]
            )
        elif len(ev.result) == 0:
            return QueryFailedEvent(
                task_name="execute_query",
                attempt=ev.attempt,
                query=ev.query,
                reason="result_empty",
                errors=[
                    "The query result is an empty list. This is most likely due to incorrect filter conditions. "
                    "Check whether the conditions use the correct nodes and properties. Another option is to relax "
                    "the filter conditions by encorporating regular expressions or alternative entity names."
                ]
            )
        else:
            # Assume that query and query result are correct
            await ctx.set("query", ev.query)
            await ctx.set("query_result", ev.result)
            return QueryResultValidatedEvent(query=ev.query, result=ev.result)

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=3, delay=5))
    async def generate_answer(
        self, ctx: Context, ev: QueryResultValidatedEvent | ClosedBookScenarioEvent | OracleScenarioEvent
    ) -> AnswerGeneratedEvent:
        llm: CustomLLM = await ctx.get("answer_llm")
        prompt_variables = {
            "question": await ctx.get("question"),
            "schema": await ctx.get("schema"),
        }
        if isinstance(ev, ClosedBookScenarioEvent):
            # Generate answer without query result in closed-book scenario
            response = await llm.achat(**ClosedBookAnswerGenerationTask.as_chat_kwargs(prompt_variables))
        else:
            # Generate answer based on query result otherwise
            response = await llm.achat(**AnswerGenerationTask.as_chat_kwargs({
                **prompt_variables, "query": ev.query, "result": ev.result
            }))
        return AnswerGeneratedEvent(
            task_name="generate_answer",
            raw=response.message.content,
            **response.additional_kwargs,
        )

    @step
    async def finish(self, ctx: Context, ev: AnswerGeneratedEvent | RetriesExceededEvent) -> StopEvent:
        """Combine all results to a dictionary"""
        keys = ("tasks", "sample_id", "question", "query", "query_result")
        result = {key: await ctx.get(key, default=None) for key in keys}
        result["answer"] = ev.raw if isinstance(ev, AnswerGeneratedEvent) else None
        return StopEvent(result=result)
