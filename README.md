# 🌱 Plant Disease Detection System

An end-to-end **machine learning project** for detecting plant diseases using environmental and soil parameters.  
This project demonstrates the complete **data science lifecycle** — from data analysis and model training to business intelligence visualization and deployment via a Streamlit web application.

---

## 📌 Overview

Plant diseases significantly affect agricultural productivity.  
This project aims to predict the presence of plant disease using **ensemble and boosting machine learning models** trained on structured agricultural data.

The system includes:
- Data analysis using Jupyter Notebook
- Visual analytics with Power BI
- Real-time and batch predictions using Streamlit

---

## 🎯 Objectives

- Perform exploratory data analysis (EDA)
- Train and compare multiple machine learning models
- Select the best-performing model
- Visualize insights using Power BI
- Deploy an interactive prediction dashboard

---

## 🧠 Machine Learning Models

The following models were trained and evaluated:

- Random Forest  
- Bagging Classifier  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

Each trained model is saved as a `.joblib` file for efficient reuse and deployment.

---

## 📊 Project Components

### 📓 Jupyter Notebook
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training and evaluation  
- Model comparison and selection  

### 📈 Power BI Dashboard
- Dataset summary and statistics  
- Feature distribution analysis  
- Disease presence trends  
- Interactive filters and visuals  

### 🌐 Streamlit Web Application
- Interactive prediction interface  
- Single-record disease prediction  
- Batch prediction via CSV upload  
- Probability-based prediction output  
- Downloadable prediction results  

---

## 🗂️ Project Structure


PLANT_DISEASE_DETECTION/
│
├── .ipynb_checkpoints/
├── catboost_info/
│
├── notebook/
│ └── plant_disease.ipynb
│
├── powerbi/
│ └── plant Disease Analysis Dashboard.pbix
│
├── app.py
├── plant_disease_dataset.csv
│
├── bagging_model.joblib
├── catboost_model.joblib
├── gradient_boosting_model.joblib
├── lightgbm_model.joblib
├── random_forest_model.joblib
├── xgboost_model.joblib
│
└── README.md


---

## 🛠️ Technologies Used

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Business Intelligence:** Power BI  
- **Web Framework:** Streamlit  
- **Models:** Ensemble and Boosting Algorithms  

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

   pip install -r requirements.txt

### 2️⃣ Run the Streamlit Application
   streamlit run app.py

Ensure the dataset and .joblib model files are present in the same directory as app.py.

---


## 📥 Input Features

- Temperature
- Humidity
- Rainfall
- Soil pH

---

### 📤 Output

- Disease Prediction: Disease / No Disease
- Prediction Probability (where applicable)

---

## 🎯 Use Cases

- Smart agriculture decision support systems
- Early plant disease detection
- Machine learning model benchmarking
- Data science portfolio project

---

## ⭐ Key Highlights

- End-to-end machine learning pipeline
- Multiple model comparison in a single project
- Integrated analytics and deployment
- Clean, modular, and scalable architecture

---

## 🔮 Future Enhancements

- Image-based disease detection using deep learning
- Integration with real-time weather APIs
- Model explainability using SHAP or LIME
- Cloud-based deployment

## 👤 Author

Zeel Panchal
Data Science & Machine Learning Enthusiast