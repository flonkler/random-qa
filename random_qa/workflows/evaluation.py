import json

from llama_index.core.workflow.retry_policy import ConstantDelayRetryPolicy
from llama_index.core.workflow import Event, StopEvent, Context, step

from random_qa.llm import CustomLLM, CorrectnessGradingTask
from random_qa.eval import BERTScorerWrapper, evaluate_exact_match, evaluate_keyword_recall, evaluate_query_results
from random_qa.workflows.base import BaseWorkflow, SetupDoneEvent


class EvaluationResultEvent(Event):
    metric: str
    score: float | dict | None


class EvaluationWorkflow(BaseWorkflow):
    query_metrics = ("recall", "precision", "f1")
    lexical_metrics = ("exact_match", "keyword_recall")
    llm_as_judge_metrics = ("correctness",)
    bert_score_metrics = ("bert_score_recall", "bert_score_precision", "bert_score_f1")

    @step
    async def evaluate_query(self, ctx: Context, ev: SetupDoneEvent) -> EvaluationResultEvent:
        """Compute recall, precision and F1-score of the generated query result"""
        ground_truth_query_result = await ctx.get("ground_truth_query_result", default=None)
        generated_query_result = await ctx.get("generated_query_result", default=None)
        # Only compute metrics if ground truth is available, otherwise set scores to `None`
        if ground_truth_query_result is not None:            
            scores = evaluate_query_results(generated_query_result, ground_truth_query_result)
        else:
            scores = [None] * len(self.query_metrics)
        
        for metric, score in zip(self.query_metrics, scores):
            ctx.send_event(EvaluationResultEvent(metric=f"query_{metric}", score=score))

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=4, delay=2))
    async def evaluate_answer_correctness(self, ctx: Context, ev: SetupDoneEvent) -> EvaluationResultEvent:
        """Let an LLM judge the factual correctness of the generated answer"""
        ground_truth_answer = await ctx.get("ground_truth_answer", default=None)
        generated_answer = await ctx.get("generated_answer", default=None)
        if ground_truth_answer is None:
            return EvaluationResultEvent(metric="answer_correctness", score=None)
        elif generated_answer is None or generated_answer == "":
            return EvaluationResultEvent(metric="answer_correctness", score={
                "score": 1,
                "justification": "Empty answer",
            })
        
        llm: CustomLLM = await ctx.get("evaluator_llm")
        prompt_variables = {
            "question": await ctx.get("question"),
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer
        }
        response = await llm.achat(**CorrectnessGradingTask.as_chat_kwargs(prompt_variables))
        try:
            # Extract fields from structured output (JSON response)
            data = json.loads(response.message.content)
            # Validate score
            score = int(data["score"])
            if score > 5 or score < 1:
                print(f"Score not between 1 and 5, retrying sample {await ctx.get('sample_id')}")
                raise ValueError("Score must be an integer between 1 and 5")
            return EvaluationResultEvent(
                metric=f"answer_{self.llm_as_judge_metrics[0]}", score={
                    "score": score,
                    "justification": data["justification"],
                    "prompt_tokens": response.additional_kwargs.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": response.additional_kwargs.get("usage", {}).get("completion_tokens")
                })
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise ValueError("Response does not satisfy the expected JSON format")
        
    @step
    async def evaluate_answer_bert_score(self, ctx: Context, ev: SetupDoneEvent) -> EvaluationResultEvent:
        ground_truth_answer = await ctx.get("ground_truth_answer", default=None)
        generated_answer = await ctx.get("generated_answer", default=None)
        bert_scorer: BERTScorerWrapper = await ctx.get("bert_scorer")
        if ground_truth_answer is None:
            scores = [None] * len(self.bert_score_metrics)
        elif generated_answer is None or generated_answer == "":
            scores = [0.0] * len(self.bert_score_metrics)
        else:
            scores = await bert_scorer.ascore(generated_answer, ground_truth_answer)            

        for metric, score in zip(self.bert_score_metrics, scores):
            ctx.send_event(EvaluationResultEvent(metric=f"answer_{metric}", score=score))
        
    @step
    async def evaluate_answer_lexical_metrics(self, ctx: Context, ev: SetupDoneEvent) -> EvaluationResultEvent:
        """Evaluate answer with heuristic-based metrics"""
        ground_truth_answer = await ctx.get("ground_truth_answer", default=None)
        generated_answer = await ctx.get("generated_answer", default=None)
        for metric, func in zip(self.lexical_metrics, (evaluate_exact_match, evaluate_keyword_recall)):
            if ground_truth_answer is None:
                score = None
            elif generated_answer is None or generated_answer == "":
                score = 0.0
            else:
                score = func(generated_answer, ground_truth_answer)

            ctx.send_event(EvaluationResultEvent(metric=f"answer_{metric}", score=score))
   
    @step
    async def finish(self, ctx: Context, ev: EvaluationResultEvent) -> StopEvent | None:
        """Collect all metric results and construct the output"""
        num_metrics = (
            len(self.query_metrics) + len(self.lexical_metrics) + len(self.llm_as_judge_metrics) + 
            len(self.bert_score_metrics)
        )
        results: list[EvaluationResultEvent] | None = ctx.collect_events(ev, [EvaluationResultEvent] * num_metrics)
        if results is None:
            return None
        
        return StopEvent(result={"sample_id": await ctx.get("sample_id"), **{r.metric: r.score for r in results}})
