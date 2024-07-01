import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to process image and find contours
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=0)

    # Find contours
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return image, contours

# Streamlit UI
def main():
    st.title('Object Counting App')
    st.sidebar.title('Options')

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process the uploaded file
        image_path = "uploaded_image." + uploaded_file.name.split(".")[-1]
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the image and find contours
        image, contours = process_image(image_path)

        # Display original image
        st.image(image, caption='Original Image', use_column_width=True)

        # Display processed image with contours
        image_with_contours = image.copy()
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        st.image(image_with_contours, caption='Image with Contours', use_column_width=True)

        # Display the number of coins detected
        st.write(f"Number of objects detected: {len(contours)}")

if __name__ == '__main__':
    main()
