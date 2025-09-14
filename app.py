import streamlit as st
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import os

# All the plant species in the trained dataset
labels = ["Neem", "Mango", "Tulsi", "Banyan", "Peepal"]

# Function to predict the species of the plant based on the trained model and input image
def prediction(img_path, model):
    # Image pre-processing
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = img.reshape(-1, 224, 224, 3)
    img = img.astype('float32')/255.0

    lst = model.predict(img)
    index = np.argmax(lst, axis=-1)[0]
    return index, lst[0][index]

# Applying mask (optional for noisy background)
def apply_mask(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (86, 255, 255))
    result = img.copy()
    result[mask==0] = (255,255,255)
    cv2.imwrite("result.jpg", result)
    return result

# Streamlit UI
def main():
    st.write("## Indian Leaf Classifier")
    image_file = st.file_uploader("Upload a leaf image (one leaf per image)")

    if st.button("Predict"):
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, caption="Uploaded Image", width=200)

            with open(image_file.name, "wb") as f:
                f.write(image_file.getbuffer())

            model = keras.models.load_model("Indian-Leaf-CNN.h5")
            result, acc = prediction(image_file.name, model)

            if acc < 0.5:
                st.warning("Low accuracy! Trying with mask...")
                img2 = apply_mask(image_file.name)
                st.image(img2, caption="Masked Image", width=200)
                result, acc = prediction("result.jpg", model)

            st.success(f"This leaf is: {labels[result]} (Accuracy: {round(acc*100,2)}%)")
        else:
            st.warning("Please upload an image.")

if __name__ == "__main__":
    main()
