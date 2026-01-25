"""
Classification Verifier for Conflict Detection

Implements reward functions for the classification agent (Task 1).

Paper Method:
  - MatchScore: Measures grouping quality based on pairwise agreement
  - Structural penalty: Penalizes invalid outputs
  - r_c = MatchScore - λ_c * Invalid
"""

import json
import re
from typing import Dict, Any
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
        "Invalid Classification JSON after all fallbacks. "
        f"Errors: {errors}"
    )


def parse_classification_output(output: str) -> Dict[int, Any]:
    """
    Parse classification agent output.

    Expected format:
    {
        "groups": {
            "0": [0, 1, ...],  // Text unit IDs in group 0
            "1": [2, 3, ...],  // Text unit IDs in group 1
            ...  // and so on
        }
    }

    Args:
        output: Raw agent output string

    Returns:
        Parsed classification dict
    """
    # Try to extract JSON
    # output = strip_think_block(output)
    # output = strip_markdown_fence(output)
    # json_match = re.search(r'\{[\s\S]*\}', output)
    # if not json_match:
    #     raise ValueError("No JSON found in classification output")
    # parsed = parse_json_with_fallback(json_match.group())
    try:
        json_str = extract_last_json(output)
        parsed = parse_json_with_fallback(json_str)
    except ValueError as e:
        raise ValueError(f"No JSON found in Classification output: {e}")  

    # Validate structure
    if "groups" not in parsed:
        raise ValueError(f"Missing 'groups' key in Classification output: {parsed}!")

    groups = parsed["groups"]

    int_groups = {}
    for k, v in groups.items():
        if not isinstance(k, (int, str)):
            raise ValueError(f"Key in Classification output is not an int-like object: {k}")
        if not isinstance(v, (list, str)):
            raise ValueError(f"Value in Classification output is not an list-like object: {v}")

        try:
            int_k = int(k)
        except ValueError:
            raise ValueError(f"Cannot convert Classification group_id to int: {k}")

        if isinstance(v, str):
            try:
                int_v = json.loads(v)
            except ValueError:
                raise ValueError(f"Cannot convert Classification unit_id_list to int: {v}")
        else:
            int_v = v

        # Ensure all elements in the list are integers
        if not isinstance(int_v, list):
            raise ValueError(f"Group value must be a list: {int_v}")

        try:
            int_v = [int(item) for item in int_v]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert all unit IDs to integers in group {k}: {int_v}, error: {e}")

        if int_k in int_groups:
            raise ValueError(f"Duplicate group_id in Classification output: {k}")

        int_groups[int_k] = int_v

    return int_groups


def compute_classification_reward(
    predicted_groups,
    ground_truth_conflicts,
    n_units: int,
    lambda_c: float = 0.5,
    alpha: float = 3.0,
    beta: float = 0.8,
    eps: float = 0.1
) -> float:
    """
    Compute classification reward based on cohesion and separation.

    Args:
        predicted_groups: Predicted grouping {"0": [0, 1, ...], "1": [2, 3, ...], ...}
        ground_truth_conflicts: Conflict labels [[0, 1], [2, 3], ...]
        n_units: Number of units
        lambda_c: Penalty weight for invalid outputs

    Returns:
        Reward between [0, 1]
    """
    # Check for structural errors
    invalid_penalty = 0.0
    if len(predicted_groups) == 0:
        # Null results have no rewards
        return 0.0
    pred_group_num = len(predicted_groups)

    # Check for missing units and invalid (out-of-range) units
    # missing_num = 0.0
    invalid_num = 0.0
    total_num = 0.0
    pred_unit_ids = set()
    for group_id in predicted_groups:
        for unit_id in predicted_groups[group_id]:
            total_num += 1.0
            # Check if unit_id is out of valid range
            if unit_id < 0 or unit_id >= n_units:
                invalid_num += 1.0
            else:
                pred_unit_ids.add(unit_id)

    # for uid in range(n_units):
    #     if uid not in pred_unit_ids:
    #         missing_num += 1.0

    if total_num == 0:
        return 0.0
    
    # Penalize both missing units and invalid (out-of-range) units
    # invalid_penalty += 0.5 * (missing_num / n_units) + 0.5 * (invalid_num / total_num)
    invalid_penalty += invalid_num / total_num
    validity = 1.0 - (invalid_penalty * lambda_c)

    total_intragroup_pairs = 0
    false_pairs = 0
    true_conflict_pairs = set(tuple(sorted(p)) for p in ground_truth_conflicts)
    for group in predicted_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total_intragroup_pairs += 1
                if tuple(sorted((group[i], group[j]))) not in true_conflict_pairs:
                    false_pairs += 1

    if total_intragroup_pairs > 0:
        purity = math.exp(-alpha * (false_pairs / total_intragroup_pairs))
    else:
        purity = 1.0

    # Support multi-group assignment: track which groups each conflict pair appears in
    conflict_group_matches = []  # Each element is a set of group IDs where this conflict pair appears

    if len(ground_truth_conflicts) > 0:
        for conflict_pair in ground_truth_conflicts:
            u1, u2 = conflict_pair[0], conflict_pair[1]

            matched_groups = set()
            for gid, groups in predicted_groups.items():
                if u1 in groups and u2 in groups:
                    matched_groups.add(gid)

            conflict_group_matches.append(matched_groups)

        # Cohesion: fraction of conflict pairs that are captured by at least one group
        matched_conflicts = sum(1 for groups in conflict_group_matches if len(groups) > 0)
        cohesion = matched_conflicts / len(ground_truth_conflicts)

        # Separation: penalize if different conflict pairs have identical group distributions
        # This encourages different conflicts to be separated into different groups
        # if matched_conflicts <= 1:
        #     separation = 1.0
        # else:
        #     # Compare group sets between conflict pairs that were actually matched
        #     matched_group_sets = [groups for groups in conflict_group_matches if len(groups) > 0]

        #     same = 0
        #     total = 0
        #     for i, groups_a in enumerate(matched_group_sets):
        #         for j, groups_b in enumerate(matched_group_sets):
        #             if i >= j:
        #                 continue
        #             total += 1
        #             # Two conflict pairs are "in the same group" if they have identical group sets
        #             if groups_a == groups_b:
        #                 same += 1

        #     separation = 1 - same / total if total > 0 else 1.0

        # match_score = cohesion * separation
        # match_score = cohesion * (eps + (1 - eps) * separation)
        match_score = cohesion * purity


        # if pred_group_num > len(ground_truth_conflicts):
        #     groupnum_penalty = math.exp(-beta * (pred_group_num - len(ground_truth_conflicts)))
        #     match_score = match_score * groupnum_penalty
        # if pred_group_num == 1:
        #     match_score = 0

    else:
        # if pred_group_num == 1:
        #     match_score = 1.0
        # else:
        #     match_score = 0.0
        match_score = 1.0
        

    # Apply penalty
    # final_reward = max(0.0, match_score - lambda_c * invalid_penalty)
    final_reward = match_score * validity

    return final_reward


# predicted_groups = json.loads("""{"groups": {"0": [0, 1, 7, 3, 4, 8],
#          "1": [2, 5, 6],
#          "2": [9]
#      }
# }""")['groups']

# ground_truth_conflicts = [[1, 7], [3, 4]]

# compute_classification_reward(predicted_groups, ground_truth_conflicts, n_units=10)