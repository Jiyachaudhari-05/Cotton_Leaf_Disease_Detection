import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import uuid

# Constants
PRED_HISTORY_FILE = "prediction_history.csv"
IMAGE_DIR = "images"

# Ensure the image directory exists
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Load prediction history from CSV
def load_prediction_history():
    if os.path.exists(PRED_HISTORY_FILE):
        return pd.read_csv(PRED_HISTORY_FILE)
    else:
        return pd.DataFrame(columns=["prediction", "image_path"])

# Save prediction history to CSV
def save_prediction_history(prediction, image_file):
    history = load_prediction_history()
    # Use a unique filename by combining UUID with the original filename
    unique_image_name = f"{uuid.uuid4()}_{image_file}"
    image_path = os.path.join(IMAGE_DIR, unique_image_name)
    
    # Save the image if it doesn't already exist
    with open(image_path, "wb") as f:
        f.write(test_image.getbuffer())
    
    new_entry = pd.DataFrame({"prediction": [prediction], "image_path": [image_path]})
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv(PRED_HISTORY_FILE, index=False)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = Image.open(test_image)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((128, 128))  # Resize to the input size expected by the model
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return predictions[0]  # Return the prediction probabilities for each class

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Prediction History"])

# Initialize prediction history in session state if it doesn't exist
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []  # To store tuples of (prediction, image)

# Main Page
if app_mode == "Home":
    st.header("LEAF DISEASE RECOGNITION SYSTEM")
    image_path = "plant2.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(""" 
    Welcome to the Leaf Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown(""" 
        #### About Dataset
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. 
        The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    st.write("You can use the camera to take a photo. If your device has multiple cameras (front and back), you can choose which one to use in your browser settings.")
    
    # Allow users to upload an image or take a photo with the camera
    uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("Or take a photo with your camera:")

    # Use the camera image if it's available, otherwise use the uploaded image
    if camera_image:
        test_image = camera_image
    elif uploaded_image is not None:
        test_image = uploaded_image
    else:
        test_image = None

    if test_image is not None:
        # Show the selected image
        st.image(test_image, caption="Selected Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        st.snow()  # Fun snow effect
        st.write("Our Prediction")
        
        # Perform prediction
        predictions = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['healthy', 'unhealthy']
        healthy_percentage = predictions[0] * 100  # Assuming healthy is class 0
        unhealthy_percentage = predictions[1] * 100  # Assuming unhealthy is class 1
        
        st.success(f"Prediction: {class_name[np.argmax(predictions)]}")
        st.write(f"Healthy: {healthy_percentage:.2f}%")
        st.write(f"Unhealthy: {unhealthy_percentage:.2f}%")
        
        # Save prediction history
        save_prediction_history(f"Healthy: {healthy_percentage:.2f}%, Unhealthy: {unhealthy_percentage:.2f}%", test_image.name)

# Prediction History Page
elif app_mode == "Prediction History":
    st.header("Prediction History")
    
    # Load prediction history
    prediction_history = load_prediction_history()
    
    # Show prediction history
    if not prediction_history.empty:
        for idx, row in prediction_history.iterrows():
            st.write(f"{idx + 1}. {row['prediction']}")
            image_path = row['image_path']
            # Check if the image file exists before displaying
            if os.path.isfile(image_path):
                st.image(image_path, caption=f"Image {idx + 1}", use_column_width=True)  # Display the image
            else:
                st.error(f"Image file '{image_path}' not found.")
    else:
        st.write("No predictions made yet.")
