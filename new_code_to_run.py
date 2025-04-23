import cv2
import numpy as np
import pytesseract
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from docx import Document
import os
from PIL import Image

# Streamlit page setup
st.set_page_config(layout="wide")
st.title("🚀 Smart Document Scanner and OCR")

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #ff9e2a, #ff4e00);
            color: white;
        }
        .stButton>button {
            background-color: #ff4e00;
            color: white;
            font-weight: bold;
        }
        .stTextArea>textarea {
            background-color: #f7f7f7;
            color: #333;
            border: 2px solid #ff4e00;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to display image
def plot_image(img, title="Image"):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=title, use_column_width=True)

# OCR pre-processing and extraction
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Upscale if needed
    if thresh.shape[1] < 1000:
        thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Optional: show the image being passed to Tesseract
    st.subheader("🧪 Preprocessed Image for OCR")
    plot_image(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    return pytesseract.image_to_string(thresh, config="--oem 1 --psm 6")

# Upload image
uploaded_file = st.file_uploader("📤 Upload your image here!", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    orig_h, orig_w = image.shape[:2]
    display_w = 800
    scale_factor = display_w / orig_w
    display_h = int(orig_h * scale_factor)
    resized_image = cv2.resize(image, (display_w, display_h))

    st.subheader("🎨 Draw a rectangle to select the portion of the document you want to scan!")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=3,
        background_image=Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)),
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="rect",
        key="canvas"
    )

    if canvas_result.json_data and "objects" in canvas_result.json_data:
        rect_data = canvas_result.json_data["objects"]
        if len(rect_data) > 0:
            rect = rect_data[0]
            if "left" in rect and "top" in rect and "width" in rect and "height" in rect:
                # Convert coordinates back to original image scale
                left = int(rect["left"] / scale_factor)
                top = int(rect["top"] / scale_factor)
                width = int(rect["width"] / scale_factor)
                height = int(rect["height"] / scale_factor)

                # Crop region from original image
                cropped_image = image[top:top+height, left:left+width]

                st.subheader("📄 Cropped Image")
                plot_image(cropped_image)

                scanned = cropped_image  # Skip unnecessary warping

                st.subheader("🖼️ Scanned Image")
                plot_image(scanned)

                # Extract text
                st.subheader("🔍 Extracted Text")
                text = extract_text(scanned)
                st.text_area("Extracted Text", text, height=200)

                # Save as Word
                if st.button("💾 Save as Word Document"):
                    image_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                    doc = Document()
                    doc.add_heading(f'Scanned Document: {image_name}', 0)
                    doc.add_paragraph(text)
                    output_path = f"{image_name}_scanned.docx"
                    doc.save(output_path)
                    st.success(f"✅ Document saved as: {output_path}")
            else:
                st.warning("⚠️ Please draw a rectangle to select the region.")
