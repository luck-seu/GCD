# Grouping And Detection (GCD)

Code Repository for "Effective Knowledge Conflict Detection via Bi-level Agentic Optimizaition"

## Overview

This project implements a lightweight multi-agent collaborative framework for detecting knowledge conflicts between textual units. The system uses a two-stage approach:

1. **Grouping Stage**: Groups textual units by topic
2. **Detection Stage**: Identifies conflicting pairs within groups

---

## File Structure

```
MARTI/
├── verifiers/
│   └── conflict/
│       ├── Classification_verifier.py  # Grouping verifier
│       └── detection_verifier.py       # Detection verifier
├── worlds/
│   └── workflows/
│       ├── conflict_workflow.py                      # Standard two-stage workflow
│       ├── conflict_workflow_1step.py                # Single-stage workflow
└── cli/
    └── configs/
        ├── conflict_detection.yaml
        └── conflict_detection_1step.yaml
```

---

## Training
for 2-step RL training:
```
bash scripts/train_conflict_2step_17b.sh
```
for 1-step RL training:
```
bash scripts/train_conflict_1step_17b.sh
```
