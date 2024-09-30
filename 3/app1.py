import cv2
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables for the first frame
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_frame, gray_frame)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for drawing artistic shapes
    mask = np.zeros_like(frame)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small movements
            (x, y, w, h) = cv2.boundingRect(contour)
            # Draw a circle around detected motion
            center = (x + w // 2, y + h // 2)
            radius = int(np.sqrt(w * h) // 2)
            cv2.circle(mask, center, radius, (0, 255, 0), -1)  # Green circle

    # Blend the original frame with the mask to create an artistic effect
    artistic_display = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

    # Show the artistic display
    cv2.imshow('Artistic Display', artistic_display)

    prev_frame = gray_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('Artistic Display', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
