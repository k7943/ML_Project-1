# 📊 Student Performance Predictor

A complete end-to-end Machine Learning project that predicts a student's **math score** based on various academic and demographic inputs. This project demonstrates the full ML pipeline — from data ingestion and preprocessing to training, evaluation, and deployment using Flask.

---

## ✨ Features

- End-to-end ML pipeline: data ingestion → transformation → model training → prediction
- GridSearchCV-style hyperparameter tuning across **10 regression models**
- Final trained model integrated into a web app via Flask
- Clean modular code under `src/` for easy scalability and reuse
- Includes real-world dataset from Kaggle with EDA and preprocessing

---

## 🧰 Tech Stack

- **Python**
- **Flask** – web framework
- **pandas**, **numpy** – data manipulation
- **matplotlib**, **seaborn** – EDA & plotting
- **scikit-learn** – ML algorithms, preprocessing, metrics
- **XGBoost**, **CatBoost** – boosting regressors
- **joblib**, **dill** – model serialization

---

## 🔍 Problem Statement

To predict a student's **math score** based on various features such as gender, race/ethnicity, parental education, lunch type, and test preparation course — helping educators identify students who may need academic support.

---

## 📁 Dataset

- **Source**: [Kaggle – Student Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Shape**: 1000 rows × 8 columns
- **Target Variable**: `math_score`

### Features:
- Categorical: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`
- Numerical: `reading_score`, `writing_score`

---

## 📊 EDA Summary

- No missing or duplicate values
- Performance varies by gender, parental education level, and test preparation
- High correlation between reading/writing scores and math score

---

## 🛠️ Data Preprocessing

- **Numerical**: Scaled using `StandardScaler`
- **Categorical**: Encoded using `OneHotEncoder`
- Combined using `ColumnTransformer`
- Serialized using `joblib` and stored as `preprocessor.pkl`

---

## 🤖 Model Training & Selection

- **Models Evaluated**:
  - Linear Regression, Lasso, Ridge
  - K-Nearest Neighbors
  - Decision Tree, Random Forest
  - XGBoost, CatBoost, AdaBoost, Gradient Boosting

- **Tuning**: Manual `GridSearchCV` via custom parameter dictionary per model
- **Evaluation Metrics**: R² Score, MAE, RMSE

### ✅ Final Model: `Lasso Regression`

| Model                     | R² Score | MAE     | RMSE    |
|--------------------------|----------|---------|---------|
| **Lasso (Best)**         | 0.8806   | 4.21    | 5.39    |
| Ridge                    | 0.8805   | 4.21    | 5.39    |
| Linear Regression        | 0.8804   | 4.21    | 5.39    |
| CatBoost Regressor       | 0.8614   | 4.46    | 5.81    |
| Gradient Boosting        | 0.8610   | 4.49    | 5.82    |
| AdaBoost Regressor       | 0.8519   | 4.68    | 6.00    |
| XGBoost Regressor        | 0.8495   | 4.65    | 6.05    |
| Random Forest Regressor  | 0.8461   | 4.70    | 6.12    |
| K-Neighbors Regressor    | 0.5786   | 7.94    | 10.13   |
| Decision Tree Regressor  | 0.5492   | 8.05    | 10.47   |

---

## 🧠 Project Structure

```bash
.
├── app.py                   # Flask application entry point
├── templates/               # HTML templates
├── data/                    # Raw data (stud.csv)
├── notebook/                # EDA and model training notebooks
├── artifacts/               # Serialized model & preprocessor
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── src/
│   ├── components/          # Data ingestion, transformation, model training
│   ├── pipeline/            # train and predict pipelines
│   ├── utils.py             # Utility functions
│   └── exception.py         # Custom exception handler

```
---

## 👤 Author

**Kunal Sharma** <br>
📧 Email: [kunals7943@gmail.com](mailto:kunals7943@gmail.com) <br>
🎓 B.Tech Chemical Engineering Student at IIT Delhi <br>
💡 Passionate about Machine Learning, Full Stack Development, and Software Engineering <br>
🔗 [LinkedIn]([https://www.linkedin.com/in/YOUR-USERNAME](https://www.linkedin.com/in/kunal-sharma-112010263/)) &nbsp;|&nbsp