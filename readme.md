# 💼 AI Salary Prediction System

## 📄 PROJECT OVERVIEW
This project predicts AI-related salaries based on multiple factors such as job role, experience level, company size, and economic indicators.  

It demonstrates a **complete end-to-end Machine Learning workflow**, including:
- Advanced preprocessing using pipelines
- Model training and evaluation
- Hyperparameter tuning
- Model comparison and selection

---

## 🗂 DATASET
**Source:** Global AI Jobs Dataset  

**Description:**  
The dataset contains features such as:
- Country, Job Role, Industry
- Experience Level, Company Size
- AI Adoption Score, Economic Indicators
- Work-life balance, Job security, etc.

**Target Variable:**  
`salary_usd`

---

## 🧰 PROJECT FILES
- `ai_salary_prediction.py` → Main ML pipeline implementation  
- `Dataset/global_ai_jobs.csv` → Dataset (excluded via `.gitignore`)  
- `requirements.txt` → Project dependencies  
- `README.md` → Project documentation  

---

## 🔧 KEY TECHNIQUES USED

### 📊 DATA PREPROCESSING
- One-Hot Encoding for nominal categorical features  
- Ordinal Encoding for ordered features (experience level, company size)  
- Log Transformation for skewed data (`bonus_usd`)  
- Feature Scaling using `StandardScaler` (for linear models only)  

---

### 🤖 MODELING
- Linear Regression  
- Support Vector Machine (LinearSVR)  
- Decision Tree Regressor  
- Random Forest Regressor  

---

### ⚙️ HYPERPARAMETER TUNING
- GridSearchCV (SVM, Decision Tree)  
- RandomizedSearchCV (Random Forest)  
- K-Fold Cross Validation  

---

### 📈 EVALUATION METRICS
- R² Score  
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  

---

## 📊 MODEL COMPARISON
All models are evaluated **before and after tuning**, and compared based on performance.

---

## 🏆 BEST MODEL
The project automatically identifies the best model based on **highest R² Score**.

---

## 💡 LEARNING OUTCOMES
- Built an **end-to-end ML pipeline using Pipeline & ColumnTransformer**
- Learned difference between **linear and tree-based models**
- Applied **feature engineering and transformation techniques**
- Implemented **hyperparameter tuning with cross-validation**
- Compared models and selected the best one

---

## 🚀 FUTURE IMPROVEMENTS
- Feature importance analysis  
- Add Gradient Boosting / XGBoost  
- Model deployment using Flask or FastAPI  
- Add SHAP for model interpretability  
- Save and load trained models  

---

## 🛠 TECH STACK
- Python  
- pandas, NumPy  
- scikit-learn  

---

## 📁 PROJECT STRUCTURE
AI-Salary-Prediction/
├── Dataset/
│   └── global_ai_jobs.csv
├── ai_salary_prediction.py
├── requirements.txt
├── README.md

---

## 👤 AUTHOR
**Inam Ur Rehman**  
BS Computer Engineering  
Focus: Machine Learning | Deep Learning | AI Engineering