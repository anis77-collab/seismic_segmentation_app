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
import tempfile
import zipfile

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
    filenames = []
    for uploaded_file in uploaded_files:
        # Read image file as numpy array and convert to RGB format
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Normalize pixel values to range [0, 1]
        image = image.astype(np.float32) / 255.0
        images_data.append(image)
        filenames.append(uploaded_file.name)
    df = pd.DataFrame({'Images': images_data, 'Filenames': filenames})
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
    augmented_images = [transform(image=img, mask=img)["image"] for img in images]
    
    # Normalize pixel values
    return augmented_images

def save_segmented_images(predictions, filenames):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "segmented_images.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for i, img in enumerate(predictions):
                # Normalize the segmented image to range [0, 255] and convert to uint8
                img = (img * 255).astype(np.uint8)
                img_name = os.path.splitext(filenames[i])[0] + "_segmented.png"
                img_path = os.path.join(tmpdir, img_name)
                cv2.imwrite(img_path, img)
                zipf.write(img_path, arcname=img_name)
        with open(zip_path, "rb") as zip_file:
            return zip_file.read(), zip_path

def main():
    st.title("Image Segmentation App")

    uploaded_files = st.sidebar.file_uploader(
        "Choose images",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        key="file_uploader",
    )

    if uploaded_files:
        data_df = images_to_dataframe(uploaded_files)
        images = data_df['Images']
        filenames = data_df['Filenames']
        preprocessed_images = preprocess_images(images)

        custom_objects = {'jaccard_loss': segmentation_models.losses.jaccard_loss}
        loaded_model = load_model('model_architecture.h5', custom_objects=custom_objects)
        loaded_model.load_weights('model_weights.h5')
        predictions = loaded_model.predict(np.array(preprocessed_images))

        plot_images(preprocessed_images, predictions)

        if st.sidebar.button("Save Segmented Images"):
            zip_data, zip_path = save_segmented_images(predictions, filenames)
            st.sidebar.download_button(
                label="Download Segmented Images",
                data=zip_data,
                file_name="segmented_images.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
