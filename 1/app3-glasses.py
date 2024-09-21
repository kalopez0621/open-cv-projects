import cv2
import numpy as np

# Paths to the DNN model files"res10_300x300_ssd_iter_140000_fp16.caffemodel"
modelFile =   r"C:\Users\kalop\Documents\open-cv-projects\1\res10_300x300_ssd_iter_140000.caffemodel" # Path to caffemodel file
configFile = r"C:\Users\kalop\Documents\open-cv-projects\1\deploy.prototxt"  # Path to prototxt file

# Load the DNN model for face detection
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the image
image = cv2.imread('Emely1.jpg')  # Replace with your image path
if image is None:
    print("Error: Could not load image.")
    exit()

h, w = image.shape[:2]

# Preprocess the image for DNN face detection
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
net.setInput(blob)

# Run the forward pass to get the face detections
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    # Only consider detections with high confidence (you can adjust the threshold)
    if confidence > 0.7:  # Adjust this value as needed
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Calculate the width and adjust height based on a 4:3 ratio
        faceWidth = endX - startX
        faceHeight = int(faceWidth * 4 / 3)  # 4:3 ratio for the face height

        # Center the face region vertically
        newStartY = max(0, startY)  # Ensure it doesn't go above the image
        newEndY = newStartY + faceHeight

        # Ensure the newEndY doesn't exceed image height
        if newEndY > h:
            newEndY = h
            newStartY = newEndY - faceHeight

        # Draw a rectangle around the face
        cv2.rectangle(image, (startX, newStartY), (endX, newEndY), (255, 0, 0), 2)

# Display the result
cv2.imshow('DNN Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()