import string
import re

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


NON_RELEVANT_TOKENS = set(stopwords.words("english")) | set(string.punctuation)
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "hundred": 100, "thousand": 1000
}


def _normalize_string(text: str) -> str:
    # Remove leading and trailing white-space
    text = text.strip()
    # Replace consecutive white-space characters to a single one
    text = re.sub(r"(\s+)", " ", text)
    # Transform string to lower-case only
    return text.lower()


def _get_tokens_multiset(text: str) -> Counter[str]:
    """Construct a multiset of tokens for a given text. To achieve better comparability between multisets, tokens are
    normalized to lower-case and filtered (i.e, stop-words and punctuations are not added to the output).
    
    Args:
        text: Sentence or text in English language
        
    Returns:
        Counter object with the tokens and how often they occured in the input
    """
    multiset = Counter()
    # Decompose string into tokens
    for token in word_tokenize(text):
        # Ignore tokens that are stop-words or punctuation characters
        if token in NON_RELEVANT_TOKENS:
            continue
        # Use numerical representation instead of number words (e.g., "four" becomes "4")
        # NOTE: Due to tokenization, composite number words (e.g., "two thousand") will not be parsed correctly
        if token in NUMBER_WORDS:
            token = str(NUMBER_WORDS[token])
        # Insert token into the multiset
        multiset.update([token])
    return multiset


def evaluate_keyword_recall(candidate: str, reference: str, use_multisets: bool = True) -> float:
    """Evaluate how many keywords from the reference text (ground truth) appear in the candidate text (prediction)
    
    Args:
        candidate: Predicted/generated text that should be evaluated
        reference: Ground truth text that is used as a reference
        use_multisets: If set to `True`, the number of occurences is considered for each token. Otherwise, the tokens
            are treated as regular sets (i.e., unique tokens are only counted once).

    Returns:
        Coverage of keywords from `reference` that appear in `candidate`
    """
    candidate, reference = _normalize_string(candidate), _normalize_string(reference)
    if candidate == reference:
        # NOTE: If both strings only consist of stop-words (e.g., "No") the metric result will be 0 even when the
        # strings are identical. To avoid this, 1 is returned in case of an exact match before tokenization.
        return 1.0
    
    multiset_cand = _get_tokens_multiset(candidate)
    multiset_ref = _get_tokens_multiset(reference)
    if not use_multisets:
        # Transform multisets into regular sets (i.e., all elements have the count 1)
        multiset_cand = Counter(set(multiset_cand.elements()))
        multiset_ref = Counter(set(multiset_ref.elements()))

    # Compute true positives and false negatives
    tp = (multiset_cand & multiset_ref).total()
    fn = (multiset_ref - multiset_cand).total()
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0


def evaluate_exact_match(candidate: str, reference: str) -> float:
    """Evaluate if the candidate text and reference text match exactly. The strings are normalized to make the
    comparison case-insensitive and ignore white-spaces."""
    return 1.0 if _normalize_string(candidate) == _normalize_string(reference) else 0.0
