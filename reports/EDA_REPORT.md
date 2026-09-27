# EDA Report — CICIDS2017

- Rows: 692,703
- Columns: 79
- Target column: `Label`

## Class Distribution

| Class | Count | % |
|---|---|---|
| BENIGN | 440,031 | 63.524% |
| DoS Hulk | 231,073 | 33.358% |
| DoS GoldenEye | 10,293 | 1.486% |
| DoS slowloris | 5,796 | 0.837% |
| DoS Slowhttptest | 5,499 | 0.794% |
| Heartbleed | 11 | 0.002% |

## Data Quality

- Columns with missing values: 1
- Columns with infinite values: 2
- Duplicate rows: 81,909 (11.825%)

## Leakage-Prone Columns (flagged for Day 3-4 review)

- None detected by name-based heuristic.

## Highly Correlated Feature Pairs (>= threshold)

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| Total Fwd Packets | Subflow Fwd Packets | 1.000 |
| Total Length of Fwd Packets | Subflow Fwd Bytes | 1.000 |
| Fwd Packet Length Mean | Avg Fwd Segment Size | 1.000 |
| Total Backward Packets | Subflow Bwd Packets | 1.000 |
| Bwd Packet Length Mean | Avg Bwd Segment Size | 1.000 |
| Fwd Header Length | Fwd Header Length.1 | 1.000 |
| Fwd PSH Flags | SYN Flag Count | 1.000 |
| Total Length of Bwd Packets | Subflow Bwd Bytes | 1.000 |
| Bwd Header Length | Subflow Bwd Packets | 1.000 |
| Total Backward Packets | Bwd Header Length | 1.000 |
| Total Fwd Packets | Fwd Header Length.1 | 1.000 |
| Fwd Header Length | Subflow Fwd Packets | 1.000 |
| Fwd Header Length.1 | Subflow Fwd Packets | 1.000 |
| Total Fwd Packets | Fwd Header Length | 1.000 |
| Subflow Fwd Packets | act_data_pkt_fwd | 1.000 |
| Total Fwd Packets | act_data_pkt_fwd | 1.000 |
| Flow IAT Max | Fwd IAT Max | 0.999 |
| Flow Duration | Fwd IAT Total | 0.999 |
| Bwd Header Length | Fwd Header Length.1 | 0.999 |
| Fwd Header Length | Bwd Header Length | 0.999 |
| Total Backward Packets | Fwd Header Length.1 | 0.999 |
| Total Backward Packets | Fwd Header Length | 0.999 |
| Fwd Header Length | Subflow Bwd Packets | 0.999 |
| Fwd Header Length.1 | Subflow Bwd Packets | 0.999 |
| Fwd Header Length.1 | act_data_pkt_fwd | 0.999 |

## Cleaning Recommendations for Day 3-4

- Drop or transform leakage-prone identifier columns listed above before training.
- Replace infinite values (commonly in rate-based flow features) before scaling — typically via clipping or dropping affected rows.
- Address class imbalance seen in the distribution table (e.g. SMOTE or class_weight='balanced').
- Consider dropping one feature from each highly-correlated pair to reduce redundancy.