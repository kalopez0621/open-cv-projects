import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Initialize variables for the first frame
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(prev_frame, gray_frame)

    # Apply a threshold to get the binary image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the detected motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected motion
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small movements
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the original frame with detected motion
    cv2.imshow('Motion Detection', frame)

    # Update the previous frame
    prev_frame = gray_frame

    # Check for key press and window close event
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Check if the window was closed via the X button
    if cv2.getWindowProperty('Motion Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
