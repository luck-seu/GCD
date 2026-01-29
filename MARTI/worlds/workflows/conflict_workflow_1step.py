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
    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the think directly now.\n</think>\n\n"

    workflow_args = kwargs.get("workflow_args", {})

    # Parse input
    textual_units = json.loads(prompt)
    ground_truth = json.loads(label)

    n_units = len(textual_units)

    if len(agents) != 1:
        raise ValueError("Conflict Workflow(1 stage) requires exactly 1 agent: [detector]")
    detector_agent = agents[0]

    # Initialize trajectory storage
    trajectory = []

    textual_units_str = ""
    for i, unit in enumerate(textual_units):
        textual_units_str += f"TEXT {i}: {unit}\n\n"

    # =================================================================
    # Stage 1: Detection Agent (Task 1)
    # =================================================================

    detection_instruction = workflow_args.get("detection_instruction", "")
    if detection_instruction:
        logger.info("Using custom detection prompt from workflow_args")
    else:
        logger.error("No custom detection prompt found")
        raise ValueError("Detection prompt must be provided in workflow_args")

    detection_instruction = detection_instruction.format(texts_str=textual_units_str)

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
        compute_global_reward
    )
    try:
        detection_result = parse_detection_output(detection_output)
        # print("Detection Result:", detection_result)
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
        detection_result = []
        global_reward = 0.0
        Precision = 0.0
        Recall = 0.0

    # Store in trajectory
    trajectory.append({
        "turn_id": 0,
        "agent_index": 0,
        "agent_name": detector_agent["agent_id"],
        "agent_role": detector_agent["agent_role"],
        "agent_input": detection_input,
        "agent_output": detection_output,
        "metadata": {},
        "agent_reward": global_reward
    })

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "final_reward": global_reward,
        "precision": Precision,
        "recall": Recall,
        "detector_result": detection_result,
    }
