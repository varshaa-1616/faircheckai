# ⚖️ FairCheck AI

### Bias Detection & Fairness Evaluation Toolkit

---

## 🚀 Overview

FairCheck AI is an interactive web application that helps detect **hidden bias in datasets and machine learning model predictions**.

As AI systems increasingly make critical decisions (hiring, loans, healthcare), ensuring fairness is essential. This tool allows users to **analyze, understand, and mitigate bias before deployment**.

---

## 🎯 Problem Statement

Machine learning models often learn from historical data that may contain **systematic bias**.
If unchecked, these models can **amplify discrimination** across groups such as gender, race, or socioeconomic status.

---

## 💡 Solution

FairCheck AI provides:

* 📊 **Dataset Bias Analysis**
* 🤖 **Model Prediction Fairness Evaluation**
* 🧠 **Human-readable explanations of metrics**
* ⚠️ **Bias risk detection (Low / Medium / High)**
* 🛠️ **Personalized mitigation suggestions**

---

## 🧩 Features

### 🔹 1. Dataset Bias Detection

* Upload CSV dataset
* Select target and sensitive column
* Detect imbalance in outcomes across groups

### 🔹 2. Model Fairness Evaluation

* Upload predictions CSV
* Analyze fairness using metrics like:

  * Selection Rate
  * Demographic Parity Difference
  * Equalized Odds Difference
  * False Negative Rate

### 🔹 3. Explainable AI

* Converts complex fairness metrics into **simple language**
* Shows **group-wise differences clearly**

### 🔹 4. Risk Assessment

* LOW / MEDIUM / HIGH bias classification
* Deployment recommendation

### 🔹 5. Personalized Suggestions

* Data-level fixes
* Model-level fixes
* Group-specific recommendations

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Libraries:** Scikit-learn, Fairlearn
* **Data Processing:** Pandas, NumPy

---

## 📂 Project Structure

```
faircheck-ai/
│
├── app.py                  # Main Streamlit app
├── adult_income_sample.csv # Sample dataset
├── predictions.csv        # Sample predictions file
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/faircheck-ai.git
cd faircheck-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📊 Usage Guide

### Step 1: Dataset Bias

* Upload dataset (e.g., `adult_income_sample.csv`)
* Select:

  * Target column → `income`
  * Sensitive column → `sex`

### Step 2: Model Fairness

* Upload predictions file (`predictions.csv`)
* Select:

  * True label → `income`
  * Prediction → `prediction`
  * Sensitive → `sex`

---

## 📈 Example Output

* Group-wise bias report
* Fairness metrics
* Bias risk classification
* Actionable recommendations

---

## 🌍 Impact (SDGs)

* **SDG 10:** Reduced Inequalities
* **SDG 8:** Decent Work and Economic Growth
* **SDG 3:** Good Health and Well-being
* **SDG 16:** Peace, Justice and Strong Institutions

---

## 🔮 Future Enhancements

* Bias mitigation (before vs after comparison)
* Visualization dashboards
* Multi-sensitive attribute analysis
* Automated fairness reports (PDF download)
* Deployment on cloud (GCP / Firebase / HuggingFace Spaces)

---

## 🤝 Contribution

Feel free to fork the repo and contribute improvements.

---

## 📌 Author

Built as part of **Google Solution Challenge** 🚀

---
