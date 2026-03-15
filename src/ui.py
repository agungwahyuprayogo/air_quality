# Import the required libraries.
import streamlit as st
import requests

from PIL import Image

import utils

# Constant variables.
# PATH_CONFIG = "../config/config.yaml"
PATH_IMAGE = "/home/assets/header_images.webp"
PATH_CONFIG = "/home/config/config.yaml"

config = utils.load_config()

# Load images in the header.
header_images = Image.open(PATH_IMAGE)
st.image(header_images)

# Add some information about the service.
st.title("Air Quality Prediction")
st.subheader("Just enter the input below then click Predict button :sunglasses:")

# Create the input form.
with st.form(key = "air_data_form"):
    # Create select box for column stasiun.
    stasiun = st.selectbox(
        label = "1.\tFrom which station is this data collected?",
        options = (
            "DKI1 (Bunderan HI)",
            "DKI2 (Kelapa Gading)",
            "DKI3 (Jagakarsa)",
            "DKI4 (Lubang Buaya)",
            "DKI5 (Kebon Jeruk) Jakarta Barat"
        )
    )
    
    # apa hayo
    # Create box for number input.
    range_pm10 = config["range_pm10"]
    pm10 = st.number_input(
        label = "2.\tEnter pm10 Value:",
        min_value = range_pm10[0] + 1,
        max_value = range_pm10[1],
        help = f"Value range from {range_pm10[0]+1} to {range_pm10[1]}"
    )

    range_pm25 = config["range_pm25"]
    pm25 = st.number_input(
        label = "3.\tEnter pm25 Value:",
        min_value = range_pm25[0] + 1,
        max_value = range_pm25[1],
        help = f"Value range from {range_pm25[0]+1} to {range_pm25[1]}"
    )

    range_so2 = config["range_so2"]
    so2 = st.number_input(
        label = "4.\tEnter so2 Value:",
        min_value = range_so2[0] + 1,
        max_value = range_so2[1],
        help = f"Value range from {range_so2[0]+1} to {range_so2[1]}"
    )

    range_co = config["range_co"]
    co = st.number_input(
        label = "5.\tEnter co Value:",
        min_value = range_co[0] + 1,
        max_value = range_co[1],
        help = f"Value range from {range_co[0]+1} to {range_co[1]}"
    )

    range_o3 = config["range_o3"]
    o3 = st.number_input(
        label = "6.\tEnter o3 Value:",
        min_value = range_o3[0] + 1,
        max_value = range_o3[1],
        help = f"Value range from {range_o3[0]+1} to {range_o3[1]}"
    )

    range_no2 = config["range_no2"]
    no2 = st.number_input(
        label = "7.\tEnter no2 Value:",
        min_value = range_no2[0] + 1,
        max_value = range_no2[1],
        help = f"Value range from {range_no2[0]+1} to {range_no2[1]}"
    )

    # Create button to submit the form.
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted.
    if submitted:
        # Create dict of all data in the form.
        raw_data = {
            "stasiun": stasiun,
            "pm10": pm10,
            "pm25": pm25,
            "so2": so2,
            "co": co,
            "o3": o3,
            "no2": no2
        }

        # Create a loading animation while predicting.
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://air-quality-api-1:8080/predict", json=raw_data).json()

        # Parse the prediction.
        if res["error_msg"] != "":
            st.error("Error occurs while predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "BAIK":
                st.error("Predicted Air Quality: TIDAK BAIK.")
            else:
                st.success("Predicted Air Quality: BAIK.")

                
