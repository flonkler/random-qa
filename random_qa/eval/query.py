import contextlib
import functools
import itertools
import json
import numpy as np

from collections import Counter
from typing import Any, Iterator

def _pairwise_mappings(
    keys_left: set, keys_right: set, ignore_matching: bool = True
) -> Iterator[tuple[tuple[str | None, str | None], ...]]:
    """Compute every possible mapping between pairs of keys from two sets. The number of mappings is determined by N!
    where N is the size of the bigger and K the size of the smaller set, respectively. In case the keys from one set are
    exhausted the remaining keys of the other set are mapped to `None`.

    Parameters:
        keys_left: Set of keys
        keys_right: Set of keys
        ignore_matching: If set to `True`, matching keys (i.e., pair with the exact same name) will not be mapped to
            other keys. This can significantly reduce the search space.

    Yields: Each possible pairwise mapping of keys at a time

    Examples:
    ```
    pairwise_mappings(set("AB"), set("BC"))
    # output: (("B", "B"), ("A", "C"))
    pairwise_mappings(set("AB"), set("BC"), ignore_matching=False)
    # output: (("A", "B"), ("B", "C")), (("B", "B"), ("A", "C"))
    pairwise_mappings(set("ABC"), set("DE"))
    # output: (("A", "E"), ("B", "D"), ("C", None)), (("A", "E"), ("C", "D"), ("B", None)),
    #         (("B", "E"), ("A", "D"), ("C", None)), (("B", "E"), ("C", "D"), ("A", None)),
    #         (("C", "E"), ("A", "D"), ("B", None)), (("C", "E"), ("B", "D"), ("A", None))
    ```
    """
    if ignore_matching:
        # Exclude keys that match exactly from the sets and store them separately
        matching_keys = keys_left & keys_right
        keys_left = keys_left - matching_keys
        keys_right = keys_right - matching_keys
    else:
        matching_keys = set()

    # Assume that left set is larger than the right set
    if len(keys_left) < len(keys_right):
        keys_left, keys_right = keys_right, keys_left
        invert_order = True
    else:
        invert_order = False

    matching_pairs = map(lambda key: (key, key), matching_keys)
    # Convert to ordered list and fill with `None` to match length of `keys_left`
    keys_right = list(keys_right) + [None] * (len(keys_left) - len(keys_right))
    for permutation in itertools.permutations(keys_left):
        if invert_order:
            non_matching_pairs = zip(keys_right, permutation)
        else:
            non_matching_pairs = zip(permutation, keys_right)
        yield (*matching_pairs, *non_matching_pairs)


def _ensure_hashable(value: Any) -> Any:
    """Ensure that a value is hashable such that it can be inserted into a (multi-)set. If the value is already
    hashable, the same value will be returned. Otherwise, a hashable representation of the input will be generated.
    """
    # Serialize dictionaries and lists to JSON strings
    if isinstance(value, (dict, list)):
        # TODO: Add serialization support for special data types (e.g., datetime)
        return json.dumps(value, sort_keys=True)
    return value


def evaluate_query_results(
    result_left: list[dict[str, Any]] | None,
    result_right: list[dict[str, Any]] | None,
    use_multisets: bool = True
) -> tuple[float | None, float | None, float | None]:
    """Evaluate a query result by comparing it to a reference and computing the metrics recall, precision and F1-score

    Parameters:
        result_left: Predicted query result that will be compared against `result_right`
        result_right: Ground truth query result
        use_multisets: If set to `True`, multisets are used for the comparison. Otherwise regular sets are used, i.e.,
            unique values appear only once.

    Returns: Tuple containg the scores for recall, precision and f1_score
    """
    if result_right is None:
        # Cannot compute scores if no ground truth is available
        return None, None, None

    if result_left is None:
        # Set all scores to 0.0 if prediction is not available (this is not the same as an empty result, i.e., `[]`)
        return 0.0, 0.0, 0.0
    
    results = []
    # Extract names of keys from the query results
    keys_left = set(itertools.chain(*[record.keys() for record in result_left]))
    keys_right = set(itertools.chain(*[record.keys() for record in result_right]))

    if len(keys_left) == len(keys_right) == 0:
        # Both sets are empty so the metrics can not be computed. Nonetheless, the scores are set to 1.0 (i.e., correct
        # prediction) because the prediction is identical to the ground truth.
        # NOTE: If the ground truth is empty, the specified query is likely faulty or it is a bad example
        return 1.0, 1.0, 1.0

    # In case the length of `keys_left` and `keys_right` differs, the list of mappings will reoccuring pairs of keys.
    # Hence, the computed scores will be cached to improve the efficiency.
    @functools.lru_cache
    def compute_confusion_scores(key_left: str | None, key_right: str | None) -> tuple[int, int, int]:
        # Create multiset from the values of the respective keys. If a key is `None` the mapping did not assign
        # a partner to the other key. To ensure correct computation of FP and FN, the multiset must be empty.
        multiset_left, multiset_right = Counter(), Counter()
        if key_left is not None:
            multiset_left.update(_ensure_hashable(r[key_left]) for r in result_left)
        if key_right is not None:
            multiset_right.update(_ensure_hashable(r[key_right]) for r in result_right)

        if not use_multisets:
            # Transform multisets into regular sets (i.e., all elements have the count 1)
            multiset_left = Counter(set(multiset_left.elements()))
            multiset_right = Counter(set(multiset_right.elements()))

        # Increment counters by the number of TP, FP and FN respectively
        tp = (multiset_left & multiset_right).total()
        fp = (multiset_left - multiset_right).total()
        fn = (multiset_right - multiset_left).total()
        return tp, fp, fn

    # Iterate over all mappings of keys to find mapping with highest agreement
    for mapping in _pairwise_mappings(keys_left, keys_right):
        # Sum number of TP, FP and FN over all pairs of keys in the current mapping
        # NOTE: Casting to float here avoids numpy's behaviour of returning `nan` instead of throwing an exception when
        #       dividing by zero.
        true_positives, false_positives, false_negatives = map(float, np.sum(
            [compute_confusion_scores(key_left, key_right) for key_left, key_right in mapping],
            axis=0
        ))
        # Use initial value 0 as a fallback if a metric cannot be computed due to dividing by 0
        recall, precision, f1_score = 0.0, 0.0, 0.0
        with contextlib.suppress(ZeroDivisionError):
            recall = true_positives / (true_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives)
            f1_score = 2 * recall * precision / (recall + precision)

        # Store scores in the result list to determine the best result afterwards
        results.append((recall, precision, f1_score))
    
    # Return the result with the highest F1-score
    return max(results, key=lambda x: x[2])
