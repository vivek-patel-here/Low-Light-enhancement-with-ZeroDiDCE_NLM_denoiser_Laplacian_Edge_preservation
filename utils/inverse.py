import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image in grayscale
img = cv2.imread("utils/22.jpg", cv2.IMREAD_GRAYSCALE)

# Normalize to [0,1]
I = img / 255.0

# Compute 1 - I
inverse_img = 1 - I
output = (inverse_img * 255).astype(np.uint8)

# Display
cv2.imwrite("inverse.jpg",output)