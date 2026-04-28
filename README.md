#  Heart Disease Prediction System

##  Overview

This project is a Machine Learning-based system designed to predict the likelihood of heart disease based on clinical parameters. The goal is to assist in early detection by analyzing patient data and providing a risk prediction.

The project was developed as part of a Problem-Based Learning (PBL) initiative and is structured into three distinct phases

---

##  Dataset

* Source: Kaggle Heart Disease Dataset
* Contains medical attributes such as:

  * Age
  * Sex
  * Chest Pain Type
  * Cholesterol
  * Resting Blood Pressure
  * Maximum Heart Rate
  * Fasting Blood Sugar
  * Resting ECG
  * Exercise-Induced Angina

---

##  Project Phases

### 🔹 Phase 1: EDA & Preprocessing

* Data cleaning and handling missing values
* Feature encoding (categorical → numerical)
* Feature scaling
* Exploratory Data Analysis (EDA) using visualizations

---

### 🔹 Phase 2: Model Training & Comparison

Multiple machine learning models were trained and evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

####  Evaluation Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

After comparison, **Logistic Regression** was selected as the best-performing model due to its balanced performance and interpretability.

---

### 🔹 Phase 3: Prediction & Deployment

* Final model: Logistic Regression
* Model is used to predict heart disease (0 = No, 1 = Yes)
* Model also provides with the probability of presence of heart disease
* A simple user interface is built using Streamlit


##  Model Artifacts

The `model/` folder contains:

* `logistic_model.pkl` → Trained Logistic Regression model
* `scaler.pkl` → Feature scaling object
* `selector.pkl` → Feature selection object
* `all_features.pkl` → Feature order reference

---

##  How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Vanshi1107/PBL-repository.git
cd PBL-repository
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

##  Project Structure

```
├── PHASE-1/
│   ├──Data/
│   │   ├──heart.csv
│   │   └──heart_cleaned.csv
│   ├──EDA.ipynb
│   └──Phase 1.pdf
├── PHASE-2/
│   ├──KNN.ipynb
│   ├──SVM.ipynb
│   └──logistic_regression.ipynb
├──PHASE-3/
│   ├── model/
│   │   ├── logistic_model.pkl
│   │   ├── scaler.pkl
│   │   ├── selector.pkl
│   │   └── all_features.pkl
│   ├── Final_model.ipynb
│   ├── app.py
│   ├── predict.py
│   └── test.py
├──.gitignore
├── requirements.txt
└── README.md
```

---

##  Output

* The model predicts:

  * **0 → No Heart Disease**
  * **1 → Heart Disease**
  * **The probability of presence of heart disease**

---

##  Future Improvements

* Integrate full ML pipeline (single serialized model)
* Improve UI/UX of the application
* Deploy as a web application
* Add probability-based predictions

---

##  Technologies Used

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn
* Streamlit

---

##  Author

Developed as part of a PBL project.
- Team Members:
    * Vanshika Maheshwari
    * Priyanshi Tamta
  

---
