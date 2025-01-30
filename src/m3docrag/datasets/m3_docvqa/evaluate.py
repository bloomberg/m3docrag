# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# https://github.com/allenai/multimodalqa/blob/master/baselines/evaluate.py

import json
import argparse
import re
import string
import numpy as np
from collections import Counter
from typing import List, Set, Tuple, Union
from scipy.optimize import linear_sum_assignment
from word2number.w2n import word_to_num
from .common_utils import *

# From here through _match_numbers_if_present was originally copied from the evaluation code of DROP dataset:
# https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py

def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    #TODO: this is not included in the original drop evaluation script, we need to have our own in the end anyways.
    elif _is_word_number(text):
        return str(float(word_to_num(text)))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False



def list_em(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        return 1.0
    else:
        return 0.0


def list_f1(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, gold_answers):
    scores_for_ground_truths = []
    for gold_answer in gold_answers:
        score = metric_fn(prediction, gold_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_predictions(predictions, gold_answers, example_types=None):
    """To support multiple gold annotations, `gold_answers` should be a list,
    with each item (either a string or a list) corresponding to one valid reference answer."""
    instance_eval_results = {}
    instance_eval_results_by_types = {}
    eval_funcs = {
        "list_em": list_em,
        "list_f1": list_f1
    }
    for qas_id in gold_answers:
        ref_answers = gold_answers[qas_id]
        if qas_id not in predictions:
            print(f"Missing prediction for question {qas_id}, and all scores for this question are set to zero")
            instance_eval_results[qas_id] = {
                metric: 0.0 for metric in eval_funcs.keys()
            }
        else:
            pred_answer = predictions[qas_id]
            instance_eval_results[qas_id] = {
                metric: metric_max_over_ground_truths(
                    func, pred_answer, ref_answers
                ) for metric, func in eval_funcs.items()
            }
        if example_types is not None:
            example_type = example_types[qas_id]
            if example_type not in instance_eval_results_by_types:
                instance_eval_results_by_types[example_type] = {}
            instance_eval_results_by_types[example_type][qas_id] = instance_eval_results[qas_id]

    eval_scores = {metric: np.mean([result[metric] for result in instance_eval_results.values()]) * 100
                   for metric in eval_funcs.keys()}

    if example_types is not None:
        eval_scores_by_types = {}
        for example_type, type_instance_eval_results in instance_eval_results_by_types.items():
            eval_scores_by_types[example_type] = {
                metric: np.mean([result[metric] for result in type_instance_eval_results.values()]) * 100 for metric in eval_funcs.keys()
            }
        return eval_scores, instance_eval_results, eval_scores_by_types
    else:
        return eval_scores, instance_eval_results
    

def eval_retrieval(qid2retrieval_results, gold_examples, recall_levels=[1, 2, 4, 5, 10]):
    """
    Calculate document recall metrics per query, including recall@k for various k values,
    and compute the average recall.

    Args:
        qid2retrieval_results (dict): Dictionary mapping query IDs (qid) to retrieval results.
            The retrieval results are a list of lists, where each sublist contains 
            [doc_id, page, score], e.g., [['53e1d168375850acbaf134da744ffae0', 3, 2.3689956665039062]].
        gold_examples (list): List of ground truth examples with 'qid' and 'supporting_context'.
            Each 'supporting_context' is a list of dictionaries, 
            e.g., [{'doc_id': '8513db80c11ea439ab11eba406ec00d9', 'doc_part': 'table'}].
        recall_levels (list): List of k values for recall@k, e.g., [1, 5, 10].

    Returns:
        dict: Dictionary containing recall per query at different recall levels,
              and the average recall for each level.
    """

    # Create a mapping from query ID to the supporting context (gold standard)
    qid2supporting_context = {
        example["qid"]: example['supporting_context'] for example in gold_examples
    }

    # Initialize dictionaries to store recall at different levels
    qid2recall_at_k = {k: {} for k in recall_levels}
    avg_recall_at_k = {k: 0.0 for k in recall_levels}

    # Loop through each query and its supporting context
    for qid, supporting_context in qid2supporting_context.items():
        # Get the retrieved documents for the current query
        retrieved_results = qid2retrieval_results.get(qid, [])

        # Extract the relevant doc_ids from supporting_context
        relevant_doc_ids = {ctx['doc_id'] for ctx in supporting_context}

        # Count total relevant documents for the query
        n_relevant = len(relevant_doc_ids)

        # For each recall level, calculate how many relevant documents were retrieved
        for k in recall_levels:
            # Get the top-k results (or fewer if there are fewer results)
            top_k_results = retrieved_results[:k]

            # Ensure doc_id uniqueness in top_k_results
            top_k_doc_ids = set(result[0] for result in top_k_results)

            # Count how many relevant documents are in the top-k results
            n_relevant_retrieved_at_k = len(top_k_doc_ids.intersection(relevant_doc_ids))

            # Calculate recall@k for the current query
            recall_at_k = n_relevant_retrieved_at_k / n_relevant if n_relevant > 0 else 0.0

            # Ensure recall is between 0 and 1
            recall_at_k = min(recall_at_k, 1.0)

            # Store recall@k for the current query
            qid2recall_at_k[k][qid] = recall_at_k

    # Calculate average recall@k across all queries
    for k in recall_levels:
        avg_recall_at_k[k] = sum(qid2recall_at_k[k].values()) / len(qid2recall_at_k[k]) if qid2recall_at_k[k] else 0.0

    # Return recall@k for each query and average recall@k
    return {
        "recall_per_qid_at_k": qid2recall_at_k,
        "average_recall_at_k": avg_recall_at_k
    }


def evaluate_prediction_file(prediction_path, gold_path='/job/datasets/m3-docvqa/MMQA_dev.jsonl'):
    # predicted_answers = json.load(open(prediction_path, encoding="utf-8"))

    if isinstance(prediction_path, dict):
        raw_prediction_json = prediction_path
    else:
        raw_prediction_json = json.load(open(prediction_path, encoding="utf-8"))
    examples = read_jsonl(gold_path)
    gold_answers, answer_modalities, hop_types, question_types = {}, {}, {}, {}

    qid2predicted_answers = {}
    qid2retrieval_results = {}
    for qid, data in raw_prediction_json.items():
        qid2predicted_answers[qid] = data['pred_answer'].strip()
        qid2retrieval_results[qid] = data['page_retrieval_results']

    eval_retrieval_results = eval_retrieval(qid2retrieval_results, examples)
    print('Average recall at K')
    print(eval_retrieval_results['average_recall_at_k'])

    predicted_answers = qid2predicted_answers
    

    all_scores = {}
    all_scores['overall'] = {}
    all_scores['modalities'] = {}
    all_scores['hop_types'] = {}
    all_scores['q_types'] = {}

    all_scores['average_recall_at_k'] = eval_retrieval_results['average_recall_at_k']

    for example in examples:
        qid = example["qid"]
        # Currently we only have one ground truth answer.
        # Even if there are multiple entries in example["answers"], the whole list should be regarded as one ref answer.
        # However, our script supports evaluation with multiple ref answers.
        # So, we will use an outer bracket here to pretend we have a list of ref answers.
        gold_answer = [str(item["answer"]) for item in example["answers"]]
        gold_answers[qid] = [gold_answer]
        answer_modality = set([item["modality"] for item in example["answers"]])
        assert len(answer_modality) == 1
        answer_modalities[qid] = answer_modality.pop()
        question_types[qid] = example["metadata"]["type"]
        hop_types[qid] = "Multi-hop" if example["metadata"]["type"] in MULTI_HOP_QUESTION_TYPES else "Single-hop"

    eval_scores, instance_eval_results = evaluate_predictions(predicted_answers, gold_answers)
    print("\n\nOverall result with different metrics: ")
    for metric, value in eval_scores.items():
        print(f"{metric}: {value}")

        all_scores['overall'][metric] = value

    modality_counts = Counter(answer_modalities.values())
    _, _, eval_scores_by_modalities = \
        evaluate_predictions(predicted_answers, gold_answers, answer_modalities)
    print("\n\nEval results for different modalities:")
    for answer_modality in sorted(eval_scores_by_modalities.keys()):
        result = eval_scores_by_modalities[answer_modality]
        print(f"{answer_modality}")
        print(f"# of examples: {modality_counts[answer_modality]}")

        all_scores['modalities'][answer_modality] = {}

        for metric, value in result.items():
            print(f"{metric}: {value}")

            all_scores['modalities'][answer_modality][metric] = value

    hop_type_counts = Counter(hop_types.values())
    _, _, eval_scores_by_hop_types = evaluate_predictions(predicted_answers, gold_answers, hop_types)
    print("\n\nType\tCount\tEM\tF1")
    for hop_type in sorted(eval_scores_by_hop_types.keys()):
        result = eval_scores_by_hop_types[hop_type]
        print(f"{hop_type}\t{hop_type_counts[hop_type]}\t{result['list_em']}\t{result['list_f1']}")

        all_scores['hop_types'][hop_type] = {}

        all_scores['hop_types'][hop_type][metric] = value

    question_type_counts = Counter(question_types.values())
    _, _, eval_scores_by_qtypes = evaluate_predictions(predicted_answers, gold_answers, question_types)
    print("\n\nType\tCount\tEM\tF1")
    for question_type in sorted(eval_scores_by_qtypes.keys()):
        result = eval_scores_by_qtypes[question_type]
        print(f"{question_type}\t{question_type_counts[question_type]}\t{result['list_em']}\t{result['list_f1']}")

        all_scores['q_types'][question_type] = {}

        all_scores['q_types'][question_type][metric] = value


    # return eval_scores
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate on drop dataset")
    parser.add_argument(
        "--prediction_path",
        type=str,
        default="predictions.json",
        help="location of the prediction file",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        default="dev.json",
        help="location of the gold file",
    )
    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path)