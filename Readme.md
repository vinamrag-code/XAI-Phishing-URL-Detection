# Phishing URL Detector — XAI Framework

Implementation of the research paper:
**"An Explainable Machine Learning Framework for URL-Based Phishing Detection"**
Kashish Mehra, Katya Chadha, Vinamra Agrawal — Jaypee Institute of Information Technology, Noida

---

## Overview

A two-layer framework for detecting phishing URLs:

- **Layer 1 — ML Classification**: Random Forest + Logistic Regression
- **Layer 2 — XAI Interpretability**: SHAP (global + local) + LIME (instance-level)

---

## Project Structure

```
phishing-detector/
├── app.py
├── random_forest_model.pkl
├── logistic_regression_model.pkl
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source**: [Phishing URLs Dataset — Kaggle](https://www.kaggle.com/datasets/victusadi/phishing-urls-dataset-with-extracted-features/data)
- **Size**: 160,064 URLs (159,244 phishing, 820 legitimate)
- **Imbalance Handling**: SMOTE

---

## Installation

```bash
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Live Demo

[Add Streamlit Cloud link after deployment]

---

## Limitations

- Analyzes URL structure only, not webpage content
- Trained on a 2020 dataset — phishing patterns evolve over time
- Intended as a first-pass filter alongside other security tools

---

## Authors

| Name | Email |
|---|---|
| Kashish Mehra | mehra.kashish455@gmail.com |
| Katya Chadha | katyachadha5@gmail.com |
| Vinamra Agrawal | agrawalvinamra12@gmail.com |

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
shap
lime
matplotlib
joblib
imbalanced-learn
```