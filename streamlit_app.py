import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load model and encoders ---
@st.cache_resource
def load_model_with_encoders():
    data = joblib.load("final_model_with_encoders.pkl")
    return data['model'], data['label_encoders']

try:
    model, label_encoders = load_model_with_encoders()
except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()

# --- Categorical Lists ---
property_types = ['Flat', 'House', 'Penthouse', 'Farm House', 'Lower Portion', 'Upper Portion', 'Room']
cities = ['Islamabad', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Karachi']
provinces = ['Islamabad Capital', 'Punjab', 'Sindh']
purposes = ['For Sale', 'For Rent']
locations = sorted(label_encoders['location'].classes_.tolist())

# --- Inject Custom CSS for Alignment ---
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .big-title {
        font-size: 32px;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 18px;
        color: #34495e;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #1abc9c;
        color: white;
        border: none;
        padding: 8px 16px;
        font-size: 16px;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #16a085;
    }
</style>
""", unsafe_allow_html=True)

# --- UI Layout ---
st.markdown('<div class="big-title">🏠 Property Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fill out the details below to estimate your property price.</div>', unsafe_allow_html=True)

# --- Form Layout ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        property_type = st.selectbox("🏗️ Property Type", property_types)
        city = st.selectbox("🏙️ City", cities)
        province_name = st.selectbox("🗺️ Province", provinces)
        purpose = st.selectbox("🎯 Purpose", purposes)

    with col2:
        location = st.selectbox("📍 Location", locations)
        longitude = st.number_input("🌐 Longitude", min_value=66.842040, max_value=75.084804, value=67.001)
        baths = st.number_input("🛁 Baths", min_value=0, max_value=10, value=1)
        bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, max_value=27, value=2)
        total_area = st.number_input("📏 Total Area (sq ft)", min_value=0.0, max_value=26952.849, value=1000.0)

    st.markdown(" ")
    submitted = st.form_submit_button("🔮 Predict Price")

# --- Prediction ---
if submitted:
    try:
        # Encode categorical fields
        property_type_encoded = label_encoders['property_type'].transform([property_type])[0]
        location_encoded = label_encoders['location'].transform([location])[0]
        city_encoded = label_encoders['city'].transform([city])[0]
        province_name_encoded = label_encoders['province_name'].transform([province_name])[0]
        purpose_encoded = label_encoders['purpose'].transform([purpose])[0]

        # Input DataFrame
        input_data = pd.DataFrame([[
            property_type_encoded,
            location_encoded,
            city_encoded,
            province_name_encoded,
            longitude,
            baths,
            purpose_encoded,
            bedrooms,
            total_area
        ]], columns=[
            'property_type', 'location', 'city', 'province_name', 'longitude',
            'baths', 'purpose', 'bedrooms', 'Total_Area'
        ])

        # Predict
        prediction = model.predict(input_data)
        st.markdown(f"<h3 style='color:green;'>💰 Estimated Price: {prediction[0]:,.2f} PKR</h3>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
