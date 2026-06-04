import cv2
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread("image.jpg")

# RGB Image
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Noise Removal
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Binary Conversion
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Edge Detection (Object Detection)
edges = cv2.Canny(gray, 100, 200)

# Display Results

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gray, cmap='gray')
plt.title("Gray")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(blur, cmap='gray')
plt.title("Smoothing")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(binary, cmap='gray')
plt.title("Binary")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(edges, cmap='gray')
plt.title("Object Detection")
plt.axis("off")

plt.show()