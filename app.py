import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('image_classifier_cnn1.h5')

# Define the classes for your dataset
classes = ['building', 'forest', 'glacier', 'mountain','sea','street']

st.title('Image Classifier')

# Upload an image for classification
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = Image.open(uploaded_image)
    image = image.resize((150, 150))  # Resize to match your model's input size
    image = np.array(image)
    image = image / 255.0  # Normalize

    # Make a prediction
    predictions = model.predict(np.expand_dims(image, axis=0))
    predicted_class = classes[np.argmax(predictions)]

    # Display the prediction
    st.subheader('Prediction:')
    st.write(f'This image is classified as {predicted_class}')
