## Final Recommended Results

These are the strongest reproducible results from the current repo state.

### Table 1. Best Baseline Model Performance (Random Forest, 500 Trees)

| Dataset Split | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.9213 | 87.85% | 0.6261 | 0.6803 | 0.6521 |
| Test | 0.9202 | 87.70% | 0.6241 | 0.6675 | 0.6451 |

Source: `outputs/baseline_metrics_random_forest_500.json`

### Table 2. Character-Level 1D-CNN (Binary URL Detection)

| Dataset Split | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.9906 | 97.30% | 0.9406 | 0.8950 | 0.9172 |
| Test | 0.9900 | 97.19% | 0.9403 | 0.8886 | 0.9137 |

Source: `outputs/cnn_metrics.json`

### Table 3. Robustness Stress Test (Binary CNN)

| Evaluation Condition | ROC-AUC | Accuracy |
| :--- | :--- | :--- |
| Clean (Baseline Test) | 0.9900 | 97.19% |
| Obfuscated (Stress Test) | 0.8861 | 78.11% |
| Degradation Drop | 0.1039 | 19.08% |

Source: `outputs/stress_test_metrics.json`

### Table 4. Cross-Dataset Generalization (Binary CNN)

| Target Dataset | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dataset 2 | 0.3375 | 31.09% | 0.3511 | 0.7289 | 0.4740 |
| Dataset 3 | 0.9758 | 92.56% | 0.8069 | 0.8565 | 0.8309 |

Source: `outputs/cross_dataset_metrics.json`

### Table 5. Character-Level 1D-CNN (Multiclass Categorization)

| Evaluation Metric | Final Score |
| :--- | :--- |
| Validation ROC-AUC (OvR) | 0.9953 |
| Validation Accuracy | 96.21% |
| Validation Macro F1-Score | 0.9436 |
| Final Training Loss | 0.0847 |
| Test ROC-AUC (OvR) | 0.9952 |
| Test Accuracy | 96.17% |
| Test Macro F1-Score | 0.9411 |
| Test Weighted F1-Score | 0.9604 |

Source: `outputs/cnn_multiclass_metrics.json`
