"""
Detection Verifier for Conflict Detection

Implements reward functions for the detection agent (Task 2).

Paper Method:
  - Per-group conflict detection accuracy
  - Global reward: IoU of predicted and gold conflict pairs
  - r_d = (1/Z_d) * Σ(P_n = P_n*) - λ_d * Invalid
  - r_global = |R*_conf ∩ R̂_conf| / |R*_conf|
"""

import json
import re
from typing import Dict, Any, List, Tuple, Set
import json5
import json_repair
from itertools import combinations
import math


def strip_think_block(text: str) -> str:
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker):].strip()

def strip_markdown_fence(text):
    return re.sub(r"```[a-zA-Z]*", "", text)

def extract_last_json(text: str) -> str:
    # Find last '}'
    end = text.rfind('}')
    if end == -1:
        raise ValueError("No closing brace found")

    # Find matching '{'
    depth = 0
    start = -1
    for i in range(end, -1, -1):
        if text[i] == '}':
            depth += 1
        elif text[i] == '{':
            depth -= 1
            if depth == 0:
                start = i
                break

    if start == -1:
        raise ValueError("No matching opening brace found")

    return text[start:end + 1]

def parse_json_with_fallback(text: str):
    errors = []

    # for loader in (json.loads, json5.loads, json_repair.loads):
    for loader in (json.loads, json5.loads):
        try:
            return loader(text)
        except Exception as e:
            errors.append((loader.__name__, str(e)))

    raise ValueError(
        "Invalid Detection JSON after all fallbacks. "
        f"Errors: {errors}"
    )

def parse_detection_output(output: str) -> Dict[str, Any]:
    """
    Parse detection agent output.

    Expected format:
    {
        "conflicts": [
            [1, 2],
            [3, 5]
        ],
        "explanation": "..."
    }

    Args:
        output: Raw agent output string

    Returns:
        Parsed detection dict
    """
    # Try to extract JSON
    # output = strip_think_block(output)
    # output = strip_markdown_fence(output)
    # json_match = re.search(r'\{[\s\S]*\}', output)
    # if not json_match:
    #     raise ValueError("No JSON found in detection output")
    # parsed = parse_json_with_fallback(json_match.group())
    try:
        json_str = extract_last_json(output)
        parsed = parse_json_with_fallback(json_str)
    except ValueError as e:
        raise ValueError(f"No JSON found in detection output: {e}")  

    # Validate structure
    if "conflicts" not in parsed:
        raise ValueError("Missing 'conflicts' key")

    conflicts = parsed["conflicts"]

    if not isinstance(conflicts, list):
        raise ValueError(f"Invalid 'conflicts' format: {type(conflicts)}")

    # Validate conflict pairs
    validated_conflicts = []
    for pair in conflicts:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid conflict pair: {pair}")
        try:
            idx1, idx2 = int(pair[0]), int(pair[1])
            # Ensure smaller index first
            if idx1 < idx2:
                validated_conflicts.append([idx1, idx2])
            else:
                validated_conflicts.append([idx2, idx1])
        except (ValueError, TypeError):
            raise ValueError(f"Invalid conflict pair indices: {pair}")

    # Remove duplicates
    validated_conflicts = list(set(map(tuple, validated_conflicts)))
    validated_conflicts = [list(p) for p in validated_conflicts]

    return validated_conflicts


# def compute_detection_reward(
#     predicted_conflicts,
#     ground_truth_conflicts,
#     classification_groups: Dict,
#     lambda_d: float = 0.5
# ) -> float:
#     """
#     Compute detection reward.

#     Args:
#         predicted_conflicts: Predicted conflict pairs
#         ground_truth_conflicts: Gold conflict pairs
#         classification_groups: Current grouping (from classifier)
#         lambda_d: Penalty weight

#     Returns:
#         Reward in [0, 1]
#     """
#     # Check for structural errors
#     invalid_penalty = 0.0

#     gt_set = set(tuple(sorted(p)) for p in ground_truth_conflicts)
#     pred_set = set(tuple(sorted(p)) for p in predicted_conflicts)
#     predict_score = []
#     pre_conflict_num = len(pred_set)

#     if pre_conflict_num == 0:
#         if len(gt_set) == 0:
#             detection_score = 1.0
#         else:
#             detection_score = 0.0
#     else:
#         # invalid_count = 0.0
#         for pair in pred_set:
#             i, j = pair
            
#             pred_ingroup = False
#             for group in classification_groups.values():
#                 if i in group and j in group:
#                     pred_ingroup = True
#                     break
            
#             if not pred_ingroup:
#                 # invalid_count += 1.0
#                 # predict_score.append(0)
#                 continue

#             normalized_pair = tuple(sorted([i, j]))
#             if normalized_pair in gt_set:
#                 predict_score.append(1)
#             else:
#                 predict_score.append(0)
        
#         # invalid_penalty = invalid_count / pre_conflict_num

#         # Average over groups
#         detection_score = sum(predict_score) / len(predict_score) if predict_score else 0.0

#     # Apply penalty
#     # final_reward = max(0.0, detection_score - lambda_d * invalid_penalty)
#     final_reward = detection_score

#     return final_reward

def compute_detection_reward(
    predicted_conflicts,
    ground_truth_conflicts,
    classification_groups: Dict,
    lambda_d: float = 0.5
) -> float:
    """
    Compute detection reward.

    Args:
        predicted_conflicts: Predicted conflict pairs
        ground_truth_conflicts: Gold conflict pairs
        classification_groups: Current grouping (from classifier)
        lambda_d: Penalty weight

    Returns:
        Reward in [0, 1]
    """
    pred_set = set(tuple(sorted(p)) for p in predicted_conflicts)
    # Only keep predicted pairs that are in the same group
    pred_set = {
        (i, j) for (i, j) in pred_set
        if any(i in g and j in g for g in classification_groups.values())
    }

    gt_set = set([tuple(sorted(p)) for p in ground_truth_conflicts])

    if len(gt_set) == 0:
        # No ground truth conflicts
        # Perfect if prediction is also empty
        # return 1.0 if len(pred_set) == 0 else 0.0
        if len(pred_set) == 0:
            return 1.0
        else:
            return 0.0

    # Compute F1 score
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    Precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    Recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1_score = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

    # final reward = F1 score
    final_reward = F1_score

    return final_reward


def compute_global_reward(
    predicted_conflicts,
    ground_truth_conflicts
) -> float:
    """
    Compute global reward (joint quality).

    Paper formula:
    r_global = |R*_conf ∩ R̂_conf| / |R*_conf|

    This is essentially recall of conflict pairs.

    Args:
        predicted_conflicts: Predicted conflict pairs
        ground_truth_conflicts: Gold conflict pairs

    Returns:
        Global reward (recall) in [0, 1]
    """

    pred_set = set(tuple(sorted(p)) for p in predicted_conflicts)

    # Normalize ground truth conflicts
    gt_set = set([tuple(sorted(p)) for p in ground_truth_conflicts])

    if len(gt_set) == 0:
        # No ground truth conflicts
        # Perfect if prediction is also empty
        # return 1.0 if len(pred_set) == 0 else 0.0
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0


    # Compute F1 score
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    Precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    Recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1_score = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

    # Global reward = F1 score
    global_reward = F1_score

    return global_reward, Precision, Recall
