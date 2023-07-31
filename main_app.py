# Importing required libraries
import numpy as np
import streamlit as st
import cv2
import tensorflow_hub as hub  
import gdown  
import os
from tensorflow.keras.models import load_model

# Google Drive link for the model
gdrive_url = 'https://drive.google.com/uc?id=1C1DdXOSFy7kFhuiFN4SX6N7yD3A7_53Z'
model_path = 'final_model.h5'

# Check if the model file already exists, if not download it
if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False)

# Loading the pretrained model with custom_objects
model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})


# Defining class names
CLASS_NAMES = ['boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever', 'bedlington_terrier', 'borzoi', 'basenji', 'scottish_deerhound', 'shetland_sheepdog', 'walker_hound', 'maltese_dog', 'norfolk_terrier', 'african_hunting_dog', 'wire-haired_fox_terrier', 'redbone', 'lakeland_terrier', 'boxer', 'doberman', 'otterhound', 'standard_schnauzer', 'irish_water_spaniel', 'black-and-tan_coonhound', 'cairn', 'affenpinscher', 'labrador_retriever', 'ibizan_hound', 'english_setter', 'weimaraner', 'giant_schnauzer', 'groenendael', 'dhole', 'toy_poodle', 'border_terrier', 'tibetan_terrier', 'norwegian_elkhound', 'shih-tzu', 'irish_terrier', 'kuvasz', 'german_shepherd', 'greater_swiss_mountain_dog', 'basset', 'australian_terrier', 'schipperke', 'rhodesian_ridgeback', 'irish_setter', 'appenzeller', 'bloodhound', 'samoyed', 'miniature_schnauzer', 'brittany_spaniel', 'kelpie', 'papillon', 'border_collie', 'entlebucher', 'collie', 'malamute', 'welsh_springer_spaniel', 'chihuahua', 'saluki', 'pug', 'malinois', 'komondor', 'airedale', 'leonberg', 'mexican_hairless', 'bull_mastiff', 'bernese_mountain_dog', 'american_staffordshire_terrier', 'lhasa', 'cardigan', 'italian_greyhound', 'clumber', 'scotch_terrier', 'afghan_hound', 'old_english_sheepdog', 'saint_bernard', 'miniature_pinscher', 'eskimo_dog', 'irish_wolfhound', 'brabancon_griffon', 'toy_terrier', 'chow', 'flat-coated_retriever', 'norwich_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'english_foxhound', 'gordon_setter', 'siberian_husky', 'newfoundland', 'briard', 'chesapeake_bay_retriever', 'dandie_dinmont', 'great_pyrenees', 'beagle', 'vizsla', 'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet', 'sealyham_terrier', 'standard_poodle', 'keeshond', 'japanese_spaniel', 'miniature_poodle', 'pomeranian', 'curly-coated_retriever', 'yorkshire_terrier', 'pembroke', 'great_dane', 'blenheim_spaniel', 'silky_terrier', 'sussex_spaniel', 'german_short-haired_pointer', 'french_bulldog', 'bouvier_des_flandres', 'tibetan_mastiff', 'english_springer', 'cocker_spaniel', 'rottweiler']

CLASS_NAMES.sort()

# Configuring Streamlit app
st.title("Canine Classifier :dog:")
st.markdown("Welcome to Canine Classifier! This application uses a deep learning model to predict the breed of a dog from an image. Please upload an image file below, and the app will predict the breed of the dog.")

# Upload button for dog image
dog_image = st.file_uploader("Please upload an image file of the dog:", type=["jpg", "jpeg", "png"])

# Predict button
submit = st.button('Predict')

# Functionality for Predict button
if submit:
    # If image is uploaded
    if dog_image is not None:
        # Convert the image file to an opencv image
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image")

        # Resize the image to match model's input shape
        opencv_image = cv2.resize(opencv_image, (350,350))  # resizing to (350,350)

        # Prepare image for model prediction
        opencv_image = opencv_image / 255.0  # if you used rescaling while training
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make prediction using the model
        Y_pred = model.predict(opencv_image)

        # Display the predicted dog breed
        st.title(f"The dog breed is most likely a {CLASS_NAMES[np.argmax(Y_pred)].replace('_', ' ').title()}.")

