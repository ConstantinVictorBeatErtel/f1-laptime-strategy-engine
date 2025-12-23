import streamlit as st
import pandas as pd
import mlflow.xgboost
import xgboost as xgb # vital for loading
from sqlalchemy import create_engine

# 1. SETUP
st.set_page_config(page_title="F1 Strategy Engine", layout="wide")
st.title("🏎️ F1 Race Strategy Predictor")

# Connect to DB (Note: Port 5433 because we are outside Docker now!)
db_url = 'postgresql+psycopg2://user:password@postgres:5432/f1_data'

# ------------------------------------------------------------------------
# 🚨 PASTE YOUR RUN ID HERE 
# ------------------------------------------------------------------------
RUN_ID = "87f955d093f14ed49e65cd537f147ed5" 
# Example: RUN_ID = "d51c8bd06fb84d0ab4675697eb4b0756"

# Load Model from MLflow
# We use the tracking URI to find where the model lives
mlflow.set_tracking_uri("http://mlflow:5000")
model_uri = f"runs:/{RUN_ID}/model"

@st.cache_resource
def load_model():
    return mlflow.xgboost.load_model(model_uri)

try:
    model = load_model()
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"Could not load model. Check Run ID and MLflow. Error: {e}")

# 2. SIDEBAR CONTROLS
st.sidebar.header("Race Simulation Parameters")
driver = st.sidebar.selectbox("Select Driver", ["Max Verstappen", "Lando Norris", "Charles Leclerc"])
compound = st.sidebar.selectbox("Tire Compound", ["SOFT", "MEDIUM", "HARD"])
laps_to_sim = st.sidebar.slider("Stint Length (Laps)", 10, 50, 25)

# 3. SIMULATION ENGINE
if st.button("Predict Race Pace"):
    # Create synthetic features for the prediction
    # We simulate a stint from Lap 1 to Lap X
    
    # Map inputs to model codes (Soft=2, Medium=1, Hard=0)
    # Note: These mappings must match what we did in the Training Notebook!
    comp_map = {'SOFT': 2, 'MEDIUM': 1, 'HARD': 0}
    
    # Simple Team Mapping (Just for demo purposes)
    team_map = {'Max Verstappen': 1, 'Lando Norris': 2, 'Charles Leclerc': 0} 
    
    # Generate the data table
    stint_data = pd.DataFrame({
        'TyreLife': range(1, laps_to_sim + 1),
        'LapNumber': range(1, laps_to_sim + 1),
        'Compound_Code': [comp_map[compound]] * laps_to_sim,
        'Team_Code': [team_map.get(driver, 1)] * laps_to_sim
    })
    
    # Predict
    preds = model.predict(stint_data)
    
    # Visualize
    stint_data['Predicted Pace'] = preds
    
    st.subheader(f"Predicted Pace: {driver} on {compound}s")
    
    # Line Chart
    st.line_chart(stint_data.set_index('TyreLife')['Predicted Pace'])
    
    # Summary Metrics
    avg_pace = preds.mean()
    deg = preds[-1] - preds[0]
    
    col1, col2 = st.columns(2)
    col1.metric("Average Lap Time", f"{avg_pace:.3f} s")
    col2.metric("Total Degradation", f"+{deg:.3f} s")