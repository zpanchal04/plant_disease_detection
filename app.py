import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted # Added for explicit import
import os
import base64
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- App Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="AgriCure AI Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark background
st.markdown("""
    <style>
    .main {background-color: #000000;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stSelectbox {background-color: #333333; color: white; border-radius: 5px;}
    .sidebar .sidebar-content {background-color: #1a1a1a;}
    h1, h2, h3 {color: #4CAF50;}
    .stExpander {border: 1px solid #444444; border-radius: 5px; padding: 10px;}
    .stDataFrame, .stTable {background-color: #222222; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- Function to Check if Pipeline is Fitted (Simplified for Random Forest) ---
def is_pipeline_fitted(pipeline):
    """Checks if the Random Forest model within the pipeline is fitted."""
    model_name = "Random Forest"
    step_name = "rf" # The step name for Random Forest in the pipeline
    try:
        check_is_fitted(pipeline.named_steps[step_name])
        logger.info(f"Model '{model_name}' is fitted.")
        return True
    except NotFittedError:
        logger.error(f"Model '{model_name}' is not fitted.")
        st.error(f"Model '{model_name}' is not fitted. Please ensure the .joblib file contains a fitted pipeline.")
        return False
    except Exception as e:
        logger.error(f"Error checking if '{model_name}' is fitted: {str(e)}")
        st.error(f"Error checking if '{model_name}' is fitted: {str(e)}")
        return False

# --- Load Model (Only Random Forest) ---
@st.cache_resource
def load_models():
    """Loads only the Random Forest model pipeline."""
    models = {}
    file = "random_forest_model.joblib"
    model_name = "Random Forest"

    if os.path.exists(file):
        try:
            pipeline = joblib.load(file)
            logger.info(f"Loaded model file: {file}")
            # Use the simplified checker function
            if is_pipeline_fitted(pipeline):
                models[model_name] = pipeline
            else:
                st.warning(f"Skipping '{model_name}' as it is not fitted.")
        except Exception as e:
            logger.error(f"Failed to load model '{file}': {str(e)}")
            st.error(f"Failed to load model '{file}': {str(e)}")
    else:
        logger.warning(f"Model file '{file}' not found.")
        st.warning(f"Model file '{file}' not found. Ensure the .joblib file is in the same directory.")

    if not models:
        logger.error("No fitted Random Forest model was loaded.")
        st.error("No fitted Random Forest model was loaded. Please check 'random_forest_model.joblib'.")
    return models

models = load_models()

# --- Load Dataset ---
@st.cache_data
def load_data():
    if os.path.exists("plant_disease_dataset.csv"):
        df = pd.read_csv("plant_disease_dataset.csv")
        logger.info("Dataset 'plant_disease_dataset.csv' loaded successfully.")
    else:
        logger.error("Dataset 'plant_disease_dataset.csv' not found.")
        st.error("Dataset 'plant_disease_dataset.csv' not found.")
        df = pd.DataFrame()  # Empty dataframe fallback
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🌿 AgriCure AI")
st.sidebar.markdown("Healing Plants with Intelligence")
# Removed "Model Comparison" as it's less relevant with a single model
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Data Exploration", "🔮 Model Prediction", "📤 Upload & Predict"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("🌱🤖 AgriCure AI – Healing Plants with Intelligence 🌿💡")
    st.markdown("""
    ### Welcome to AgriCure AI Dashboard
    This advanced Streamlit dashboard is built for the Plant Disease Prediction ML project. 
    It allows you to explore the dataset, make predictions using the **Random Forest Model**, and more.
    
    **Key Features:**
    - Data visualization and statistics
    - Interactive predictions with the tuned **Random Forest** model
    - Upload your own data for batch predictions
    - Built on an ensemble ML model for high accuracy.
    
    Navigate using the sidebar to get started!
    """)
    
    # Placeholder image (replace with actual image if available)
    st.image("https://via.placeholder.com/800x300?text=AgriCure+AI+Banner", use_column_width=True)

# --- Data Exploration Page ---
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    
    if not df.empty:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        st.subheader("Visualizations")
        
        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Greens", ax=ax)
        st.pyplot(fig)
        
        # Distribution Plots
        st.markdown("#### Feature Distributions")
        cols = st.columns(2)
        for i, col in enumerate(["temperature", "humidity", "rainfall", "soil_pH"]):
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax, color="lightgreen")
            cols[i % 2].pyplot(fig)
        
        # Disease Presence Pie Chart
        st.markdown("#### Disease Presence Distribution")
        fig, ax = plt.subplots()
        df['disease_present'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'salmon'], ax=ax)
        ax.set_ylabel('') # Remove y-axis label for cleaner pie chart
        st.pyplot(fig)
    else:
        st.info("No dataset loaded. Please ensure 'plant_disease_dataset.csv' is available.")

# --- Model Prediction Page ---
elif page == "🔮 Model Prediction":
    st.title("🔮 Random Forest Model Prediction")
    
    model_name = "Random Forest"
    if model_name in models:
        st.subheader("Input Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        
        with col2:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        
        # Model selection removed as only one model is used
        st.info(f"Using the **{model_name}** model for prediction.")
        
        if st.button("Predict Disease"):
            try:
                # Prepare input
                input_data = np.array([[temperature, humidity, rainfall, soil_ph]], dtype=np.float64) # Use np.float64 for wider compatibility
                logger.info(f"Making prediction with model '{model_name}' for input: {input_data}")
                
                pipeline = models[model_name]
                prediction = pipeline.predict(input_data)[0]
                
                # Check for predict_proba attribute on the final estimator 'rf'
                if hasattr(pipeline.named_steps['rf'], 'predict_proba'):
                    # The second class (index 1) probability is the probability of disease (assuming 1 = Disease)
                    prob = pipeline.predict_proba(input_data)[0][1]
                else:
                    prob = None
                
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("⚠️ Disease Present (High Risk)")
                else:
                    st.success("✅ No Disease (Low Risk)")
                
                if prob is not None:
                    st.progress(prob)
                    st.write(f"Probability of Disease: **{prob:.2%}**")
            except NotFittedError:
                logger.error(f"Prediction failed: Model '{model_name}' is not fitted.")
                st.error(f"Model '{model_name}' is not fitted. Please use a fitted model.")
            except Exception as e:
                logger.error(f"Prediction failed for '{model_name}': {str(e)}")
                st.error(f"Error during prediction for '{model_name}': {str(e)}")
    else:
        st.info("The Random Forest model is not loaded. Please ensure 'random_forest_model.joblib' is available and fitted.")

# --- Upload & Predict Page ---
elif page == "📤 Upload & Predict":
    st.title("📤 Upload Data & Batch Predict (Random Forest)")
    
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    
    if uploaded_file:
        upload_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(upload_df.head())
        
        required_cols = ["temperature", "humidity", "rainfall", "soil_pH"]
        if set(required_cols).issubset(upload_df.columns):
            
            model_name = "Random Forest"
            if model_name in models:
                st.info(f"Using the **{model_name}** model for batch prediction.")
                
                if st.button("Run Batch Prediction"):
                    try:
                        logger.info(f"Running batch prediction with model '{model_name}'")
                        
                        # Select only the required columns and convert to np.float64
                        input_features = upload_df[required_cols].astype(np.float64) 
                        
                        predictions = models[model_name].predict(input_features)
                        
                        upload_df["Predicted_Disease_Code"] = predictions
                        upload_df["Predicted_Disease"] = upload_df["Predicted_Disease_Code"].map({0: "No Disease", 1: "Disease Present"})
                        
                        st.subheader("Prediction Results")
                        st.dataframe(upload_df)
                        
                        # Download Button
                        csv = upload_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except NotFittedError:
                        logger.error(f"Batch prediction failed: Model '{model_name}' is not fitted.")
                        st.error(f"Model '{model_name}' is not fitted. Please use a fitted model.")
                    except Exception as e:
                        logger.error(f"Batch prediction failed for '{model_name}': {str(e)}")
                        st.error(f"Error during batch prediction for '{model_name}': {str(e)}")
            else:
                 st.info("The Random Forest model is not loaded. Cannot run batch prediction.")
        else:
            st.error(f"Uploaded CSV must contain columns: {', '.join(required_cols)}")
    else:
        st.info("Please upload a CSV file for batch prediction.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ by AgriCure AI | Powered by Streamlit | Data Source: Plant Disease Dataset")