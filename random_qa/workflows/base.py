import asyncio
import functools
import neo4j
import os
from llama_index.core.workflow import Event, StartEvent, Workflow, Context, step
from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from pydantic import ConfigDict, BaseModel

from typing import get_args, get_origin, get_type_hints, Union, Any
from types import UnionType

class LLMResponseEvent(Event):
    model_config = ConfigDict(extra="ignore")  # ignore additional kwargs passed during initialization
    raw: str
    system_fingerprint: str
    model: str
    usage: dict[str, Any]
    duration: float


class TrackedEvent(Event):
    task_name: str
    attempt: int | None = None
    success: bool = True


class SetupDoneEvent(Event):
    pass


class BaseWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_tracking_step()

    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> SetupDoneEvent:
        """Store all key-value pairs from the start event in the context"""
        for k, v in ev.items():
            await ctx.set(k, v)
        return SetupDoneEvent()
    
    def _add_tracking_step(self) -> None:
        """Add a step to the workflow that saves intermediate results to the context"""
        async def store_results(ctx: Context, ev: TrackedEvent) -> None:
            # Store the attributes of the event in the workflow context
            tasks: dict[str, Any] = await ctx.get("tasks", {})
            task_id = ev.task_name + (f"_{ev.attempt}" if ev.attempt is not None else "")
            tasks[task_id] = {**tasks.get(task_id, {}), **ev.model_dump(exclude={"task_name", "attempt"})}
            await ctx.set("tasks", tasks)

        # Do not add the store_results step if it already exists in the workflow
        if store_results.__name__ in self._get_steps():
            return

        # Determine annotated return types that are subclasses of TrackedEvent
        tracked_events = set()
        for step_func in self._get_steps().values():
            # Get the return type of the step functions
            ret_types = get_type_hints(step_func)["return"]
            # Resolve union types
            if get_origin(ret_types) in (Union, UnionType):
                ret_types = get_args(ret_types)
            else:
                ret_types = (ret_types,)
            # Append classes whose bases contain the TrackedEvent class
            tracked_events.update(filter(lambda t: any(b.__name__ == "TrackedEvent" for b in t.__bases__), ret_types))
        
        if len(tracked_events) > 0:
            # Manipulate type annotations such that all the desired events are included
            store_results.__annotations__["ev"] = functools.reduce(lambda a, b: a | b, tracked_events)
            # Add function as step to the workflow
            step(store_results, workflow=self)

    async def _run_query_session(self, driver: neo4j.AsyncDriver, query: str) -> list[dict[str, Any]]:
        """Run a Cypher query in a dedicated session. This avoids conflicts between parallel queries since a new query
        cannot be started in the same session before the previous query has been consumed completly.
        
        Args:
            driver: Asynchronous driver to create a new (asynchronous) session
            query: Cypher statement that should be executed

        Returns:
            Result of the executed query
        """
        # Reference: https://neo4j.com/docs/api/python-driver/current/async_api.html#neo4j.AsyncSession
        async def inner():
            async with driver.session() as session:
                result = await session.run(query)
                return await result.data()
        
        data = await asyncio.shield(inner())
        return data
