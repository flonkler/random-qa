import asyncio
import gc
import torch.cuda
from contextlib import AbstractContextManager, AbstractAsyncContextManager
from bert_score import BERTScorer


class BERTScorerWrapper(AbstractContextManager):
    """Compute scores using a `BERTScorer` instance, supports asynchronous calls and contexts."""
    def __init__(self, model_name: str, device: str | None = None):
        """
        Args:
            model_name: Name of the transformer model. Recommended options (as of version 0.3.8) are 
                `"microsoft/deberta-xlarge-mnli"` or `"microsoft/deberta-large-mnli"` (faster).
            device: Torch device for running the model (e.g., `"cuda"` to run on GPU, `"cpu"` to run on CPU)
        """
        self._scorer = BERTScorer(
            model_type=model_name, device=device, rescale_with_baseline=True, lang="en", idf=False
        )
        self._lock = asyncio.Lock()

    def score(self, candidate: str, reference: str) -> tuple[float, float, float]:
        """Compute the BERTScore metrics for a single pair of strings."""
        scores = self._scorer.score([candidate], [reference], return_hash=False, batch_size=1)
        # Convert tensor results to tuple of floats
        return tuple(map(lambda x: float(x[0]), scores))

    async def ascore(self, candidate: str, reference: str) -> tuple[float, float, float]:
        """Compute the BERTScore metrics asynchronously."""
        async with self._lock:
            scores = await asyncio.to_thread(self.score, candidate, reference)
            return scores
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Free up memory when leaving the context
        del self._scorer
        gc.collect()    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
