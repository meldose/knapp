import cv2
import numpy as np

# Checkerboard parameters
rows = 6  # inner corners
cols = 9  # inner corners
square_size_m = 0.025  # in meters
dpi = 300  # print resolution

# Convert square size to pixels for printing
square_size_inch = square_size_m / 0.0254
square_size_px = int(square_size_inch * dpi)

# Image size
img_height = rows * square_size_px
img_width = cols * square_size_px

# Create the checkerboard
checkerboard = np.zeros((img_height, img_width), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            y_start = i * square_size_px
            x_start = j * square_size_px
            checkerboard[y_start:y_start + square_size_px, x_start:x_start + square_size_px] = 255

# Save the checkerboard
cv2.imwrite("checkerboard_9x6_25mm.png", checkerboard)
print("Checkerboard saved as 'checkerboard_9x6_25mm.png'")
