# Additional training datasets

All four CSVs are synthetic demonstration data, not captured intrusions. They use the existing project's feature schema. Synthetic results do not establish real-world accuracy or reliable attacker intent prediction.

| File | Rows | Use |
| --- | ---: | --- |
| synthetic_sequence_baseline.csv | 2,280 | Simple next-move training: 240 episodes |
| synthetic_sequence_variable.csv | 4,534 | More varied next-move training: 360 episodes |
| synthetic_sequence_evaluation.csv | 1,466 | Reserved later evaluation: 120 separate episodes; do not train on this file |
| synthetic_binary_balanced.csv | 6,000 | Binary IDS training: 3,000 normal and 3,000 attack observations |

## Upload instructions

For next-move training, open Model registry, expand **Train on labeled intrusion sequences** under **Next-move prediction model**, choose `synthetic_sequence_variable.csv`, and select **Train next-move model**. The baseline file is an alternative simpler experiment. Each upload replaces the previous trained model; uploads do not accumulate.

For normal/attack detection, use **Train with your dataset** in Model registry, choose `synthetic_binary_balanced.csv`, and select **Train models**.

Keep the evaluation CSV separate. The app does not currently offer a separate CSV evaluation upload. This file contains later timestamps and episode IDs absent from the two training files, but shares their generator assumptions.

Sequence targets are the next recorded behavior label, including possible continuation of the same behavior. They do not specify when an attack will occur. Labels cover benign traffic, reconnaissance, credential access, lateral movement, exfiltration, and denial of service.

## Provenance and validation

`dataset_manifest.json` records seeds, class counts, purposes, and SHA-256 hashes. `dataset-tools/build.mjs` reproduces the files. `dataset_validation.json` records checks against the actual backend: sequence validation, unique episode IDs across files, later evaluation timestamps, and successful training of the three training files in temporary model directories. Active model files were not replaced.

## Captured-traffic research sources

- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html

These research datasets require preprocessing before use with this project. Their columns and labels do not directly match this app. For next-move training, preserve temporal ordering and establish valid episode boundaries and behavior labels from supporting ground truth. Do not manufacture missing feature values or infer a validated attack sequence merely from adjacent rows.
