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
    from marti.verifiers.conflict.classification_verifier import (
        parse_classification_output,
        compute_classification_reward
    )
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

    classification_result_str = ""
    if isinstance(classification_result, str):
        classification_result_str += classification_result
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
    # Stage 2: Detection Agent (Task 2)
    # =================================================================

    detection_instruction = workflow_args.get("detection_instruction", "")
    if detection_instruction:
        logger.info("Using custom detection prompt from workflow_args")
    else:
        logger.error("No custom detection prompt found")
        raise ValueError("Detection prompt must be provided in workflow_args")

    detection_instruction = detection_instruction.format(classification_result=classification_result_str)

    detection_input = apply_template_with_tokenizer(
        detector_agent["tokenizer"],
        detection_instruction
    )
    detection_response = await detector_agent["llm"].generate_async.remote(
        detection_input,
        detector_agent["sampling_params"]
    )
    detection_output = detection_response.outputs[0].text
    # print("Detection Output:", detection_output)

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

    
    from marti.verifiers.conflict.detection_verifier import (
        parse_detection_output,
        compute_detection_reward,
        compute_global_reward
    )
    try:
        detection_result = parse_detection_output(detection_output)
        # print("Detection Result:", detection_result)
        try:
            detection_reward = compute_detection_reward(
                detection_result,
                ground_truth,
                classification_result,
                workflow_args.get("invalid_penalty_lambda_d", 0)
            )
        except Exception as e:
            logger.error(f"Detection reward compute error: {e}")
            detection_reward = 0.0  # Assign negative reward for invalid output

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
    except Exception as e:
        logger.error(f"Detection parsed error: {e}. Fake 0 conflict.")
        # print("Detection Output:", detection_output)
        detection_result = []
        detection_reward = 0.0
        global_reward = 0.0
        Precision = 0.0
        Recall = 0.0




    trajectory.append({
        "turn_id": 1,
        "agent_index": 1,
        "agent_name": detector_agent["agent_id"],
        "agent_role": detector_agent["agent_role"],
        "agent_input": detection_input,
        "agent_output": detection_output,
        "metadata": {},
        "agent_reward": detection_reward
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
