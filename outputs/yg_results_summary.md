# 📊 **Final Experimental Results (New Runs Only)**

---

## 🔹 1. Baseline Model — Random Forest (500 Trees)

| Split      | ROC-AUC | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| ---------- | ------- | -------- | ------------------- | ---------------- | ------------------ |
| Validation | 0.9265  | 90.08%   | 0.7431              | 0.6227           | 0.6776             |
| Test       | 0.9251  | 89.95%   | 0.7418              | 0.6132           | 0.6714             |

---

## 🔹 2. Character-Level CNN — Binary Classification

### 📈 Training Summary

* Epochs: 50
* Final Training Loss: **0.0964**
* Best Validation ROC-AUC: **0.9652**
* Best Validation Accuracy: **93.33%**
* Best Validation F1: **0.7935**

---

### 📊 Final Test Performance

| Metric              | Value  |
| ------------------- | ------ |
| ROC-AUC             | 0.9636 |
| Accuracy            | 92.98% |
| Precision (Class 1) | 0.8593 |
| Recall (Class 1)    | 0.6946 |
| F1-Score (Class 1)  | 0.7682 |

---

### 📉 Training Stability Snapshot (Selected Epochs)

| Epoch | Loss   | Val ROC-AUC | Val Accuracy | Val F1 |
| ----- | ------ | ----------- | ------------ | ------ |
| 1     | 0.1956 | 0.9444      | 91.09%       | 0.6991 |
| 10    | 0.1171 | 0.9619      | 92.92%       | 0.7669 |
| 20    | 0.1065 | 0.9639      | 92.53%       | 0.7339 |
| 30    | 0.1014 | 0.9638      | 91.89%       | 0.7759 |
| 40    | 0.0985 | 0.9651      | 92.48%       | 0.7843 |
| 50    | 0.0964 | 0.9650      | 90.94%       | 0.7640 |

---

## 🔹 3. Character-Level CNN — Multiclass Classification

### 📈 Training Summary

* Classes: benign, defacement, malware, phishing
* Final Training Loss: **0.1279**
* Best Validation ROC-AUC (Macro OvR): **0.9855**
* Best Validation Accuracy: **93.23%**
* Best Validation Macro F1: **0.9183**

---

### 📊 Final Test Performance

| Metric              | Value  |
| ------------------- | ------ |
| ROC-AUC (OvR Macro) | 0.9854 |
| Accuracy            | 92.81% |
| Macro F1-Score      | 0.9145 |
| Weighted F1-Score   | 0.9282 |

---

### 📉 Training Stability Snapshot (Selected Epochs)

| Epoch | Loss   | Val ROC-AUC | Val Accuracy | Val Macro F1 |
| ----- | ------ | ----------- | ------------ | ------------ |
| 1     | 0.3034 | 0.9771      | 91.13%       | 0.8841       |
| 10    | 0.1666 | 0.9840      | 92.78%       | 0.9105       |
| 20    | 0.1466 | 0.9848      | 92.97%       | 0.9109       |
| 30    | 0.1375 | 0.9855      | 92.86%       | 0.9150       |
| 40    | 0.1319 | 0.9852      | 93.10%       | 0.9138       |
| 50    | 0.1279 | 0.9843      | 92.97%       | 0.9139       |

---

## 🔹 4. Robustness Evaluation (Stress Test)

### 📊 Performance Comparison

| Condition       | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| --------------- | ------- | -------- | --------- | ------ | -------- |
| Clean Test      | 0.7810  | 83.12%   | 0.4951    | 0.4283 | 0.4592   |
| Obfuscated Test | 0.7522  | 72.28%   | 0.3327    | 0.6522 | 0.4406   |

---

### ⚠️ Degradation Metrics

| Metric        | Drop   |
| ------------- | ------ |
| ROC-AUC Drop  | 0.0288 |
| Accuracy Drop | 10.84% |
| F1 Drop       | 0.0186 |

---

## 🔹 5. Cross-Dataset Generalization

### 📊 Evaluation Results

| Target Dataset | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| -------------- | ------- | -------- | --------- | ------ | -------- |
| Dataset 2      | 0.3181  | 34.17%   | 0.3723    | 0.7956 | 0.5072   |
| Dataset 3      | 0.9929  | 96.98%   | 0.9211    | 0.9389 | 0.9299   |

---

