import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import os
from docx import Document
from tkinter import Tk, filedialog

# Function to select an image
Tk().withdraw()  # Prevents root window from appearing
img_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
if not img_path:
    print("No image selected!")
    exit()

# Load the image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the original image
def plot_image(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_image(img)

# Convert to grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_img = get_grayscale(img)
plot_image(gray_img, cmap='gray')

# Apply Gaussian Blur
def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 1)

blur_img = remove_noise(gray_img)
plot_image(blur_img, cmap='gray')

# Apply Canny Edge Detection
def Edge_detection(image):
    return cv2.Canny(image, 100, 200)

edged_img = Edge_detection(blur_img)
plot_image(edged_img, cmap='gray')

# Detecting Contours
contours, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

doc_cnts = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
    if len(approx) == 4:
        doc_cnts = approx
        break

if doc_cnts is None:
    print("Document outline not found!")
    exit()

# Perspective Transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

warped = four_point_transform(img, doc_cnts.reshape(4, 2))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
plot_image(warped, cmap='gray')

# OCR using Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
out_below = pytesseract.image_to_string(warped)
print(out_below)

# Save extracted text to a Word file
image_name = os.path.splitext(os.path.basename(img_path))[0]
doc_filename = f"{image_name}_scanned.docx"
doc = Document()
doc.add_paragraph(out_below)
doc.save(doc_filename)
print(f"Text saved to {doc_filename}")