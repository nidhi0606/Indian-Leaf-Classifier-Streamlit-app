import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import keras
from PIL import Image
from tempfile import NamedTemporaryFile
import cv2

# Indian plant species labels
labels = ["Neem", "Mango", "Tulsi", "Banyan", "Peepal"]

# Prediction function
def prediction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224), color_mode="grayscale")
    img = image.img_to_array(img)
    img = img.reshape(-1, 224, 224, 1)
    img = img.astype("float32") / 255.0
    lst = model.predict(img)
    index = np.argmax(lst, axis=-1)[0]
    return index, lst[0][index]

# Apply mask to filter background
def apply_mask(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    range1 = (36, 0, 0)
    range2 = (86, 255, 255)
    mask = cv2.inRange(hsv, range1, range2)
    result = img.copy()
    result[mask == 0] = (255, 255, 255)
    cv2.imwrite("result.jpg", result)
    return result

# Cache the model
@st.cache(allow_output_mutation=True)  # for Streamlit 0.80.0
def load_model():
    return keras.models.load_model("Indian-Leaf-CNN.h5")

# Main UI
def main():
    st.title("🌿 Indian Plant Leaf Classifier")
    st.write("Upload a leaf image to identify its species (Neem, Mango, Tulsi, Banyan, Peepal).")

    image_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

    if st.button("Predict"):
        if image_file is not None:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(image_file.getbuffer())
                temp_path = temp_file.name

            # Use beta_columns for Streamlit 0.80.0
            c1, c2 = st.beta_columns([1, 5])

            # Show uploaded image
            with c1:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Image", width=120)

            with c2:
                st.write("🔎 Classifying...")
                model = load_model()
                result, acc = prediction(temp_path, model)

                # If low confidence, try masked image
                if acc < 0.5:
                    st.warning("Low accuracy detected! Trying with background mask...")
                    img2 = apply_mask(temp_path)
                    with c1:
                        st.image(img2, caption="Masked Image", width=120)
                    result, acc = prediction("result.jpg", model)

                # Show result
                st.success(f"✅ Predicted species: **{labels[result]}**")
                st.info(f"Confidence: {round(acc*100, 2)}%")

                # Google search link
                search = "+".join(labels[result].split())
                st.markdown(f"[🔍 Learn more about {labels[result]}](https://www.google.com/search?q={search}+plant+leaf)")

        else:
            st.error("⚠️ Please upload an image before predicting.")

if __name__ == "__main__":
    main()
