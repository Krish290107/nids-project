# Model Report

Best model: **random_forest** (selected by `macro_f1` on the held-out test set).

## Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Train time (s) |
|---|---|---|---|---|
| random_forest | 0.9995 | 0.9974 | 0.9995 | 115.4 |
| xgboost | 0.9994 | 0.9961 | 0.9994 | 63.0 |

## Per-Class Results — random_forest

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BENIGN | 0.9999 | 0.9997 | 0.9998 | 83,407 |
| DoS GoldenEye | 0.9923 | 0.9985 | 0.9954 | 2,057 |
| DoS Hulk | 0.9994 | 0.9993 | 0.9993 | 34,570 |
| DoS Slowhttptest | 0.9905 | 0.9971 | 0.9938 | 1,046 |
| DoS slowloris | 0.9981 | 0.9935 | 0.9958 | 1,077 |
| Heartbleed | 1.0 | 1.0 | 1.0 | 2 |

## Top Features — random_forest

| Feature | Importance |
|---|---|
| Avg Bwd Segment Size | 0.10228 |
| Packet Length Variance | 0.09036 |
| Fwd IAT Total | 0.06671 |
| Init_Win_bytes_forward | 0.06121 |
| Subflow Fwd Bytes | 0.05519 |
| Fwd IAT Mean | 0.05352 |
| Destination Port | 0.04359 |
| Active Min | 0.0429 |
| Idle Min | 0.04038 |
| Bwd IAT Total | 0.03993 |
| Bwd IAT Mean | 0.0303 |
| Bwd IAT Max | 0.02859 |
| act_data_pkt_fwd | 0.02849 |
| Active Mean | 0.02455 |
| Flow Bytes/s | 0.02283 |

## Notes

- The test set was never resampled, so these numbers reflect the real class balance.
- Macro F1 weights every class equally, so a model that misses a rare attack class is penalized.
- Figures: `reports/figures/confusion_matrix_*.png`, `feature_importance.png`, `model_comparison.png`.