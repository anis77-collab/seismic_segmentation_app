
import streamlit as st
from PIL import Image
import os
from tensorflow.keras.models import load_model

import numpy as np
import albumentations as A
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import cv2
import pandas as pd


import segmentation_models.metrics
import segmentation_models.losses



# Function for model prediction
def plot_images(images_left, images_right):
    st.write("")  # Add some space for better layout
    num_images_left = len(images_left)
    num_images_right = len(images_right)
    num_columns = 2
    rows = max(num_images_left, num_images_right)
    for i in range(rows):
        cols = st.columns(2)
        if i < num_images_left:
            cols[0].image(images_left[i], caption="Uploaded Image", use_column_width=True)
        if i < num_images_right:
            cols[1].image(images_right[i], caption="Segmented Image", use_column_width=True)

def images_to_dataframe(uploaded_files):
    images_data = []
    for uploaded_file in uploaded_files:
        # Read image file as numpy array and convert to RGB format
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Normalize pixel values to range [0, 1]
        image = image.astype(np.float32) / 255.0
        images_data.append(image)
    df = pd.DataFrame({'Images': images_data})
    return df
def aug(image_size=192, crop_prob=1, train=True):
    if train:
        return A.Compose([
            A.PadIfNeeded(min_height=image_size, min_width=image_size, p=1),
            A.CropNonEmptyMaskIfExists(width=image_size, height=image_size, p=crop_prob),
            # Add other augmentations if needed
        ], p=1)
    else:
        return A.Compose([
            A.PadIfNeeded(min_height=image_size, min_width=image_size, p=1),
            A.CropNonEmptyMaskIfExists(width=image_size, height=image_size, p=crop_prob),
        ], p=1)

# Preprocess images with albumentations
def preprocess_images(images, image_size=192):
    # Define the augmentation function
    transform = aug(image_size=image_size, train=False)
    
    # Apply augmentation to each image
    augmented_images = [transform(image=img, mask = img)["image"] for img in images]
    
    # Normalize pixel values
    
    return augmented_images

def main():
    st.title("Image Segmentation App")

    uploaded_files = st.sidebar.file_uploader(
        "Choose images",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        key="file_uploader",
    )

    if uploaded_files:
        # Load and preprocess images
        #augmented_images = [aug()(image=np.array(img), mask=np.array(img))["image"] for img in images]
        #preprocessed_images = [preprocess_image(img) for img in augmented_images]
    
        # Make predictions
        #predictions = predict_images(preprocessed_images)

        # Display original and segmented images
        #st.write("Original and Segmented Images:")
        #display_images(images, predictions)
        data_df = images_to_dataframe(uploaded_files)
        #print(data_df.head(2))
        images = data_df['Images']
        preprocessed_images = preprocess_images(images)
        #print(np.array(preprocessed_images).shape)
    
        custom_objects = {'jaccard_loss': segmentation_models.losses.jaccard_loss,
                 }
        loaded_model = load_model('/workspaces/seismic_segmentation_app/model_architecture(1).h5', custom_objects=custom_objects)
        loaded_model.load_weights('/workspaces/seismic_segmentation_app/model_weights(1).h5')
        predictions = loaded_model.predict(np.array(preprocessed_images))
        num_images = predictions.shape[0]
        plot_images(np.array(preprocessed_images), predictions)

if __name__ == "__main__":
    main()

