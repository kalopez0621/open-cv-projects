import cv2
import numpy as np
import pygame

# Initialize pygame for sound
pygame.mixer.init()

# Load your sound effect
sound_effect = pygame.mixer.Sound(r'C:\Users\kalop\Documents\open-cv-projects\3\SoundEffect1.wav')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables for the first frame
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Variable to track if sound is playing
sound_playing = False

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
    motion_detected = False  # Flag to check if motion is detected

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small movements
            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            size = int(np.sqrt(w * h) // 2)

            # Draw a triangle and rectangle
            triangle_points = np.array([[center[0], center[1] - size],
                                         [center[0] - size, center[1] + size],
                                         [center[0] + size, center[1] + size]], np.int32)
            cv2.polylines(mask, [triangle_points], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue triangle
            cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

            motion_detected = True  # Set flag to true if motion is detected

    # Play sound effect if motion is detected
    if motion_detected:
        if not sound_playing:  # Check if the sound is already playing
            sound_effect.play()  # Play the sound
            sound_playing = True   # Update the flag
    else:
        if sound_playing:  # If motion stopped and sound was playing
            sound_effect.stop()  # Stop the sound
            sound_playing = False  # Update the flag

    # Blend the original frame with the mask to create an artistic effect
    artistic_display = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

    # Show the artistic display
    cv2.imshow('Artistic Display with Geometric Patterns and Sound', artistic_display)

    prev_frame = gray_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('Artistic Display with Geometric Patterns and Sound', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
