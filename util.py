"""HotpotQA answer normalization and EM/F1 metrics."""

import re
import string
from collections import Counter


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        return "".join(char for char in value if char not in set(string.punctuation))

    return " ".join(remove_articles(remove_punctuation((text or "").lower())).split())


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero_metric = (0.0, 0.0, 0.0)

    special_answers = {"yes", "no", "noanswer"}
    if normalized_prediction in special_answers and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in special_answers and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
