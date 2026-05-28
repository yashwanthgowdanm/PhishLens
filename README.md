# PhishLens: Neural Phishing & Malware URL Detection

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PhishLens is a dual-engine phishing and malicious URL detection system developed for CSE 543: Information Assurance & Security at Arizona State University. It shifts the defensive paradigm from fragile, heuristic-based lexical rules to automated spatial pattern recognition using a PyTorch Character-Level 1D Convolutional Neural Network (CNN).

The system evaluates raw URL character sequences to catch zero-day threats, structural obfuscation (like homoglyphs and delimiter injections), and parameter stuffing, achieving a 96.36% ROC-AUC on binary classification and strong resilience under adversarial stress testing.

## 🔍 Project Overview
Traditional phishing detection relies heavily on manually engineered features (e.g., keyword dictionaries, entropy), which are easily bypassed by adversaries using simple character permutations. This project explores end-to-end sequence modeling to map raw URLs into a 32-dimensional continuous space. 

**Key Capabilities:**
* **Binary & Multiclass Tracking:** Categorizes URLs as benign, phishing, malware, or defacement.
* **Adversarial Robustness:** Built-in stress testing against homoglyph substitutions, delimiter injections, and subdomain spoofing.
* **Dual-Engine Inference:** Combines the deep spatial reasoning of the 1D-CNN with a deterministic heuristic fallback to ensure real-time, low-latency threat mitigation.

## 📂 Repository Structure
```text
GROUP-12-CSE-543/
│
├── outputs/                  # Saved models (.pth), metrics (.json), and charts
├── processed_data/           # Cleaned and normalized datasets
│   ├── dataset1_binary/      # Primary binary training/testing splits
│   ├── dataset1_multiclass/  # 4-class categorization splits
│   ├── dataset2_binary/      # Unseen distribution for generalization testing
│   └── dataset3_binary/      # Unseen distribution for generalization testing
│
├── .gitignore
├── app.js                    # Frontend logic for the PhishLens operator console
├── cross_dataset_eval.py     # Tests model generalization across unseen datasets
├── data_paths.py             # Helpers for resolving dataset directories
├── index.html                # PhishLens UI layout and dashboard
├── merge_new_data.py         # Cleans, audits, and merges adversarial URL data
├── README.md                 # Project documentation
├── server.py                 # Lightweight REST API and dual-engine inference server
├── stress_test.py            # Synthetically mutates URLs to evaluate robustness
├── styles.css                # Styling for the PhishLens UI
├── train_baseline.py         # Trains traditional ML baselines (Random Forest, SVM, LogReg)
└── train_cnn.py              # Trains the PyTorch Character-Level 1D-CNN
```

## ⚙️ Prerequisites & Dependencies
The system is designed to be lightweight and deployable in resource-constrained environments. Dedicated GPU acceleration (CUDA/MPS) is supported but not strictly required for inference.

* **Language:** Python 3.9+
* **Core Libraries:**
    * `torch` (PyTorch)
    * `scikit-learn`
    * `pandas`
    * `numpy`

## 🚀 Installation & Setup
Follow these steps to set up the environment on a local Windows, macOS, or Linux machine.

```bash
# Clone the repository
git clone https://github.com/yashwanthgowdanm/PhishLens.git
cd PhishLens

# Create and activate a virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install required packages
pip install torch scikit-learn pandas numpy
```

## 💻 Usage Guide

### 1. Model Training
You can train the traditional baseline models or the neural networks from scratch. Models and metric reports are automatically saved to the `outputs/` directory.

**Train Traditional Baselines:**
```bash
# Options: logreg, random_forest, svm
python train_baseline.py --model random_forest
```

**Train the Character-Level 1D-CNN:**
```bash
# Task options: auto, binary, multiclass
python train_cnn.py --task binary --epochs 50 --batch-size 128
python train_cnn.py --task multiclass --epochs 50 --batch-size 128
```

### 2. Evaluation & Stress Testing
Evaluate the trained PyTorch models against synthetic adversarial attacks and unseen data distributions.

**Run the Robustness Stress Test:**
Applies homoglyphs, subdomain injections, and delimiter manipulations to the test set to calculate degradation deltas ($\Delta$ Accuracy and $\Delta$ AUC).
```bash
python stress_test.py
```

**Run Cross-Dataset Generalization:**
Evaluates the model trained on Dataset 1 against unseen distributions (Dataset 2 and Dataset 3).
```bash
python cross_dataset_eval.py
```

### 3. Live Inference Server (PhishLens UI)
Launch the PhishLens dual-engine server to classify URLs in real-time through the web dashboard. The server automatically loads the pre-trained `.pth` weights from the `outputs/` directory.

```bash
python server.py --host 127.0.0.1 --port 5173
```
* Open your browser and navigate to `http://127.0.0.1:5173`
* Use the Operator Console to paste URLs and toggle between **Binary Risk** and **Multiclass Family** modes.

## 🏗️ System Architecture & Key Findings

### The Neural Architecture
The `CharCNN` maps raw URLs (up to 200 characters) via an 83-character vocabulary into a 32-dimensional embedding space. A 1D Convolutional layer (128 filters, kernel size 5) captures n-gram spatial correlations, followed by Adaptive Max Pooling to ensure translation invariance (detecting malicious payloads regardless of where they are buried in the URL).

### Quantitative Highlights
* **Binary Performance:** The CNN achieved a **92.98% Accuracy** and **0.9636 ROC-AUC**, significantly outperforming the Random Forest baseline (which suffered from a low 61.32% recall on phishing URLs).
* **Multiclass Categorization:** Successfully separates Benign, Defacement, Malware, and Phishing URLs with a **Macro F1-Score of 0.9145**.
* **Robustness:** Under heavy synthetic obfuscation (stress testing), the CNN maintained an ROC-AUC of **0.7522** (dropping only 0.0288), proving its spatial representations are far more resilient than deterministic exact-match rules.
* **Generalization:** Achieved a near-perfect **0.9929 ROC-AUC** on unseen Dataset 3, proving the model learns generalized threat syntax rather than memorizing training data.
```
