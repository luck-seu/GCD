"""
Knowledge Conflict Detection Workflow

Paper: Lightweight Agent-Collaborative Framework for Knowledge Conflict Detection
Two-stage workflow:
  1. Classification Agent: Groups textual units by topic
  2. Detection Agent: Identifies conflicting pairs within groups
"""
from copy import deepcopy
import os
import asyncio
import json
from typing import Dict, List, Any, Optional
from marti.helpers.logging import init_logger
from marti.worlds.workflows.utils import apply_template_with_tokenizer
from marti.verifiers.conflict.classification_verifier import (
    parse_classification_output,
    compute_classification_reward
)
from marti.verifiers.conflict.detection_verifier import (
    parse_detection_output,
    compute_detection_reward,
    compute_global_reward
)

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


async def workflow(
    prompt: str,              # The concatenated textual units
    label: str,               # Ground truth labels (JSON string)
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Two-stage conflict detection workflow

    Args:
        prompt: JSON string containing:
            {
                "textual_units": [...],
            }
        label: JSON string containing:
            {
                "conflicts": [[1, 2], ...]
            }
        agents: List of 2 agents [classifier, detector]

    Returns:
        {
            "prompt": original prompt,
            "label": ground truth labels,
            "trajectory": [classifier_turns, detector_turns],
            "reward_matrix": [classifier_rewards, detector_rewards],
            "final_reward": global_reward
        }
    """
    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the think directly now.\n</think>\n\n"

    workflow_args = kwargs.get("workflow_args", {})

    # Parse input
    textual_units = json.loads(prompt)
    ground_truth = json.loads(label)

    n_units = len(textual_units)

    if len(agents) != 2:
        raise ValueError("Conflict Workflow requires exactly 2 agents: [classifier, detector]")
    classifier_agent = agents[0]
    detector_agent = agents[1]

    # Initialize trajectory storage
    trajectory = []

    # =================================================================
    # Stage 1: Classification Agent (Task 1)
    # =================================================================

    classification_instruction = workflow_args.get("classification_instruction", "")
    if classification_instruction:
        logger.info("Using custom classification prompt from workflow_args")
    else:
        logger.error("No custom classification prompt found")
        raise ValueError("Classification prompt must be provided in workflow_args")

    # Construct classification prompt
    textual_units_str = ""
    for i, unit in enumerate(textual_units):
        textual_units_str += f"TEXT {i}: {unit}\n\n"

    classification_instruction = classification_instruction.format(texts_str=textual_units_str)

    classifier_input = apply_template_with_tokenizer(
        classifier_agent["tokenizer"],
        classification_instruction
    )
    classifier_response = await classifier_agent["llm"].generate_async.remote(
        classifier_input,
        classifier_agent["sampling_params"]
    )
    classification_output = classifier_response.outputs[0].text

    if "<|im_end|>" not in classification_output:
        classifier_agent["sampling_params"].max_tokens = 512
        new_classification_output = ""
        if "</think>" not in classification_output:
            new_classification_input = classifier_input + classification_output + early_stopping_text
            new_classifier_response = await classifier_agent["llm"].generate_async.remote(
                new_classification_input,
                classifier_agent["sampling_params"]
            )
            new_classification_output = early_stopping_text + new_classifier_response.outputs[0].text
        else:
            new_classification_input = classifier_input + classification_output
            new_classifier_response = await classifier_agent["llm"].generate_async.remote(
                new_classification_input,
                classifier_agent["sampling_params"]
            )
            new_classification_output = new_classifier_response.outputs[0].text
        classification_output += new_classification_output

    # Parse classification outputs and compute rewards
    try:
        classification_result = parse_classification_output(classification_output)
        # print("Classification Result:", classification_result)
        try:
            classification_reward = compute_classification_reward(
                classification_result,
                ground_truth,
                n_units,
                workflow_args.get("invalid_penalty_lambda_c", 0)
            )
        except Exception as e:
            logger.error(f"Classification reward compute error: {e}")
            classification_reward = 0.0  # Assign negative reward for invalid output
    except Exception as e:
        logger.error(f"Classification parsed error: {e}. Assign all text to 1 group.")
        # print("Classification Output:", classification_output)
        classification_result = {}
        classification_result[0] = []
        for i in range(n_units):
            classification_result[0].append(i)

        classification_reward = 0.0  # Assign negative reward for invalid output

    classification_result_strs = []
    if isinstance(classification_result, str):
        classification_result_str = classification_result
        classification_result_str += "\n(Note: The above is the raw classification output.)\n"
    elif isinstance(classification_result, dict):
        unique_groups = {}
        group_signature_map = {}

        for gid, units in classification_result.items():
            valid_units = sorted(set(u for u in units if 0 <= u < n_units))
            if not valid_units:
                continue
            signature = tuple(valid_units)
            if signature not in group_signature_map:
                new_gid = len(unique_groups)
                group_signature_map[signature] = new_gid
                unique_groups[new_gid] = valid_units

        if len(unique_groups) != len(classification_result):
            logger.warning("Classification result has duplications or empty groups. Get zero reward.")
            classification_reward = 0.0

        classification_result = unique_groups

        for group_id in sorted(classification_result.keys()):
            classification_result_str = ""
            valid_unit_ids = []
            for unit_idx in classification_result[group_id]:
                # Bounds checking to prevent index out of range errors
                if 0 <= unit_idx < n_units:
                    valid_unit_ids.append(unit_idx)
                else:
                    logger.warning(f"Invalid unit index {unit_idx} in group {group_id}, skipping (valid range: 0-{n_units-1})")
            if not valid_unit_ids:
                continue
            classification_result_str += f"**GROUP {group_id}:**\n"
            for unit_idx in valid_unit_ids:
                classification_result_str += f"TEXT {unit_idx}: {textual_units[unit_idx]}\n"
            classification_result_str += "\n"
            classification_result_strs.append(classification_result_str)
    
    # Store in trajectory
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": classifier_agent["agent_id"],
        "agent_role": classifier_agent["agent_role"],
        "agent_input": classifier_input,
        "agent_output": classification_output,
        "metadata": {},
        "agent_reward": classification_reward
    })

    # =================================================================
    # Stage 2: Detection Agent (Task 2) - Async Parallel Processing
    # =================================================================

    detection_prompt = workflow_args.get("detection_instruction", "")
    if detection_prompt:
        logger.info("Using custom detection prompt from workflow_args")
    else:
        logger.error("No custom detection prompt found")
        raise ValueError("Detection prompt must be provided in workflow_args")

    async def process_detection_for_group(
        group_idx: int,
        classification_result_str: str
    ) -> Dict[str, Any]:
        """Process detection for a single group asynchronously"""
        detection_instruction = detection_prompt.format(classification_result=classification_result_str)

        detection_input = apply_template_with_tokenizer(
            detector_agent["tokenizer"],
            detection_instruction
        )
        detection_response = await detector_agent["llm"].generate_async.remote(
            detection_input,
            detector_agent["sampling_params"]
        )
        detection_output = detection_response.outputs[0].text

        if "<|im_end|>" not in detection_output:
            detector_agent["sampling_params"].max_tokens = 512
            new_detection_output = ""
            if "</think>" not in detection_output:
                new_detection_input = detection_input + detection_output + early_stopping_text
                new_detector_response = await detector_agent["llm"].generate_async.remote(
                    new_detection_input,
                    detector_agent["sampling_params"]
                )
                new_detection_output = early_stopping_text + new_detector_response.outputs[0].text
            else:
                new_detection_input = detection_input + detection_output
                new_detector_response = await detector_agent["llm"].generate_async.remote(
                    new_detection_input,
                    detector_agent["sampling_params"]
                )
                new_detection_output = new_detector_response.outputs[0].text
            detection_output += new_detection_output

        return {
            "group_idx": group_idx,
            "detection_input": detection_input,
            "detection_output": detection_output
        }

    # Create async tasks for all groups
    detection_tasks = [
        process_detection_for_group(i, classification_result_str)
        for i, classification_result_str in enumerate(classification_result_strs)
    ]

    # Execute all detection tasks concurrently
    detection_group_results = await asyncio.gather(*detection_tasks)

    # Parse and merge all detection outputs
    all_detection_conflicts = []
    for result in detection_group_results:
        try:
            detection_conflicts = parse_detection_output(result["detection_output"])
            all_detection_conflicts.extend(detection_conflicts)
        except Exception as e:
            logger.error(f"Detection parsed error for group {result['group_idx']}: {e}")
            all_detection_conflicts.extend([])
            # Continue processing other groups

    # Remove duplicates from merged conflicts
    all_detection_conflicts = list(set(map(tuple, all_detection_conflicts)))
    all_detection_conflicts = [list(p) for p in all_detection_conflicts]

    detection_result = all_detection_conflicts

    # Compute rewards using merged detection results
    try:
        detection_reward = compute_detection_reward(
            detection_result,
            ground_truth,
            classification_result,
            workflow_args.get("invalid_penalty_lambda_d", 0)
        )
    except Exception as e:
        logger.error(f"Detection reward compute error: {e}")
        detection_reward = 0.0

    try:
        global_reward, Precision, Recall = compute_global_reward(
            detection_result,
            ground_truth,
        )
    except Exception as e:
        logger.error(f"Global reward compute error: {e}")
        global_reward = 0.0
        Precision = 0.0
        Recall = 0.0

    # Add all detection group turns to trajectory
    for i, result in enumerate(detection_group_results):
        trajectory.append({
            "turn_id": 1 + i,
            "agent_index": 1,
            "agent_name": detector_agent["agent_id"],
            "agent_role": detector_agent["agent_role"],
            "agent_input": result["detection_input"],
            "agent_output": result["detection_output"],
            "metadata": {"group_idx": result["group_idx"]},
            "agent_reward": detection_reward / len(detection_group_results) if detection_group_results else 0.0
        })

    # all_rewards = [classification_reward, detection_reward]
    # for turn, reward in zip(trajectory, all_rewards):
    #     turn["agent_reward"] = reward

    final_reward = global_reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": final_reward,
        "precision": Precision,
        "recall": Recall,
        "classifier_result": classification_result,
        "detector_result": detection_result,
    }
