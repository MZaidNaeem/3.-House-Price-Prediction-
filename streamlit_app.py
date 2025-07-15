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

# --- UI Options ---
property_types = ['Flat', 'House', 'Farm House', 'Lower Portion', 'Upper Portion', 'Room']
cities = ['Islamabad', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Karachi']
provinces = ['Islamabad Capital', 'Punjab', 'Sindh']
purposes = ['For Sale', 'For Rent']
locations = sorted(label_encoders['location'].classes_.tolist())  # full list from trained encoder

# --- Streamlit UI ---
st.title("🏠 Property Price Prediction")

with st.form("prediction_form"):
    st.header("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        property_type = st.selectbox("Property Type", property_types)
        city = st.selectbox("City", cities)
        province_name = st.selectbox("Province", provinces)
        purpose = st.selectbox("Purpose", purposes)

    with col2:
        location = st.selectbox("Location", locations)
        longitude = st.number_input("Longitude", min_value=66.842040, max_value=75.084804, value=67.001)
        baths = st.number_input("Number of Baths", min_value=0, max_value=10, value=1)
        bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=27, value=2)
        total_area = st.number_input("Total Area (sq ft)", min_value=0.0, max_value=26952.849, value=1000.0)

    submitted = st.form_submit_button("Predict Price")

# --- Prediction Logic ---
if submitted:
    try:
        # Encode categorical fields using loaded encoders
        property_type_encoded = label_encoders['property_type'].transform([property_type])[0]
        location_encoded = label_encoders['location'].transform([location])[0]
        city_encoded = label_encoders['city'].transform([city])[0]
        province_name_encoded = label_encoders['province_name'].transform([province_name])[0]
        purpose_encoded = label_encoders['purpose'].transform([purpose])[0]

        # Create input dataframe
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
        st.success(f"💰 Predicted Property Price: {prediction[0]:,.2f} PKR")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
