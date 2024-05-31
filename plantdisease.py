import streamlit as st
from PIL import Image
import requests
import io
import tempfile
import datetime

from roboflow import Roboflow

# Function to make API request to Roboflow model
def get_prediction(image_path):
    # Replace 'YOUR_MODEL_API_ENDPOINT' with the actual API endpoint of your Roboflow model
    rf = Roboflow(api_key="4hcMhArALJwWNtDdJ5DF")
    project = rf.workspace().project("plant-disease-detector")
    model = project.version(1).model

    # infer on the local image file
    result = model.predict(image_path, confidence=1, overlap=40).json()

    # visualize your prediction
    prediction_image = model.predict(image_path, confidence=1, overlap=40).save("prediction.jpg")

    return result, prediction_image

def get_disease_severity(prediction_result):
    # Replace this with your logic to calculate severity based on the prediction result
    # For example, you might use the number or size of detected disease objects
    severity = len(prediction_result.get("objects", [])) * 10  # Adjust as needed
    return severity

# Function to display seasonal care calendar based on the current month
def display_seasonal_calendar(month):
    seasonal_care_info = {
        "January": "Winter is here! Take this time to assess your soil quality. Consider adding organic matter to improve soil structure.",
        "February": "Continue preparing your soil for the upcoming planting season. Check for signs of pests in stored garden supplies.",
        "March": "Spring is here! Start soil testing for nutrient levels. Watch for early signs of pests and take preventive measures.",
        "April": "Assess and amend your soil based on the soil test results. Set up barriers or traps for pests like slugs and snails.",
        "May": "Continue soil improvement practices. Mulch around plants to retain moisture and deter weeds. Keep an eye on aphids and caterpillars.",
        "June": "Summer is approaching. Water deeply and regularly to maintain soil moisture. Introduce beneficial insects to control pests.",
        "July": "Mid-summer care is crucial for soil health. Consider cover cropping to add nutrients. Monitor for signs of soil-borne diseases.",
        "August": "Continue monitoring soil health. Remove spent crops to prevent disease. Introduce natural predators for pest control.",
        "September": "Fall is around the corner. Test soil pH and adjust if needed. Clean up garden debris to reduce hiding places for pests.",
        "October": "Prepare your soil for winter by adding organic matter. Control overwintering pests by cleaning up the garden and removing hiding spots.",
        "November": "Fall is a great time for soil enrichment. Consider cover cropping to protect and improve the soil over winter.",
        "December": "Winter is here. Protect your soil from erosion by using cover crops. Plan for spring soil amendments and pest prevention.",
        # Add more months and corresponding care information
    }

    if month in seasonal_care_info:
        st.subheader(f"Seasonal Care for {month} is:")
        st.write(seasonal_care_info[month])
    else:
        st.warning("No seasonal care information available for this month.")


def main():
    st.title("Plant Disease Detector")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        image_path = temp_image.name
        image = Image.open(uploaded_file)
        image.save(image_path, format="JPEG")

        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Make prediction
        result, prediction_image = get_prediction(image_path)
        
        #st.write("Prediction Result:", result)
        image_pred = Image.open("C:/Plant_disease/prediction.jpg")
        st.image(image_pred, caption="Prediction Image.", use_column_width=True)
        class_to_detect = result['predictions'][0]['class']
        if class_to_detect == 'Healthy leaf':
            image_path = "C:\Plant_disease\prediction.jpg"
            st.write("It is a healthy leaf")
            current_month = datetime.datetime.now().strftime("%B")
            st.subheader(f"Current Month: {current_month}")
            display_seasonal_calendar(current_month)
        else:
            image_path = "C:\Plant_disease\prediction.jpg"
            severity_percentage = disease_severity_assessment(image_path)
            if 0 <= severity_percentage <= 20:
                severity = "Very Mild"
            elif 20 < severity_percentage <= 40:
                severity = "Mild"
            elif 40 < severity_percentage <= 60:
                severity = "Severe"
            else:
                severity = "Very Severe"
            #print(f"Disease Severity: {severity}")

            st.write(f"Disease Severity: {severity}")
            current_month = datetime.datetime.now().strftime("%B")
            st.subheader(f"Current Month: {current_month}")
            display_seasonal_calendar(current_month)

        # image_path = "C:\Plant_disease\prediction.jpg"
        # severity_percentage = disease_severity_assessment(image_path)
        # print(f"Disease Severity: {severity_percentage:.2f}%")
        #st.write("The severity of the disease is:", severity_percentage)

        # Close the temporary file
        temp_image.close()


import cv2
import numpy as np
from skimage import measure

def disease_severity_assessment(image_path):
    # Load the image
    original_image = cv2.imread(image_path)

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    _, thresh = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate disease severity
    total_area = grayscale_image.shape[0] * grayscale_image.shape[1]
    disease_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust the threshold as needed
            disease_area += area

    severity_percentage = (disease_area / total_area) * 100

    return severity_percentage

if __name__ == '__main__':
    main()


