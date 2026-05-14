import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Nassau Candy Route Efficiency", layout="wide")
st.title("🍬 Nassau Candy Distributor")
st.subheader("Factory-to-Customer Shipping Route Efficiency Predictor")

@st.cache_resource
def load_model():
    model = joblib.load('shipping_efficiency_model.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("📦 Shipment Details")

distance = st.sidebar.number_input("Distance (km)", min_value=10.0, max_value=800.0, value=150.0)
traffic = st.sidebar.slider("Traffic Index (1=Low, 10=High)", 1.0, 10.0, 5.0)
weather = st.sidebar.slider("Weather Score (1=Poor, 10=Excellent)", 1.0, 10.0, 7.0)
weight = st.sidebar.number_input("Shipment Weight (kg)", min_value=5.0, max_value=2000.0, value=300.0)
stops = st.sidebar.number_input("Number of Stops", min_value=0, max_value=7, value=2)
experience = st.sidebar.number_input("Driver Experience (years)", min_value=0.5, max_value=20.0, value=5.0)

vehicle = st.sidebar.selectbox("Vehicle Type", metadata['categorical_categories']['vehicle_type'])
time = st.sidebar.selectbox("Time of Day", metadata['categorical_categories']['time_of_day'])
day = st.sidebar.selectbox("Day of Week", metadata['categorical_categories']['day_of_week'])
route = st.sidebar.selectbox("Route Type", metadata['categorical_categories']['route_type'])

input_data = pd.DataFrame({
    'distance_km': [distance],
    'traffic_index': [traffic],
    'weather_score': [weather],
    'shipment_weight_kg': [weight],
    'num_stops': [stops],
    'driver_experience_yrs': [experience],
    'vehicle_type': [vehicle],
    'time_of_day': [time],
    'day_of_week': [day],
    'route_type': [route]
})

if st.sidebar.button("Predict Efficiency Score"):
    prediction = model.predict(input_data)[0]
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.metric("📊 Predicted Route Efficiency Score", f"{prediction:.1f} / 100")
        if prediction >= 80:
            st.success("✅ Highly Efficient Route")
        elif prediction >= 60:
            st.info("⚡ Moderately Efficient Route")
        elif prediction >= 40:
            st.warning("⚠️ Low Efficiency – Consider optimization")
        else:
            st.error("❌ Very Inefficient – Immediate action needed")
    st.markdown("---")
    st.subheader("📈 What influences this score?")
    st.write("""
    - **Distance** – longer routes reduce efficiency
    - **Traffic & Weather** – high traffic and poor weather lower score
    - **Shipment weight & stops** – heavier loads and more stops reduce efficiency
    - **Driver experience** – experienced drivers improve efficiency
    - **Route type** – highways are best, urban/rural may be slower
    """)

st.markdown("---")
st.caption("ML model trained on historical shipment data | For Nassau Candy Distributor internal use")
