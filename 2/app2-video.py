import cv2
import numpy as np

# Function to filter the video stream based on current slider values
def filter_color(frame, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the specific color range
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create the mask
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Apply the mask to filter the original video frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    return result

# Callback function for the trackbars (does nothing, but required by createTrackbar)
def nothing(x):
    pass

def main():
    # Open the webcam (0 is the default camera, change if you have multiple cameras)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Create a window named "Dynamic Color Filtering"
    cv2.namedWindow("Dynamic Color Filtering", cv2.WINDOW_NORMAL)

    # Create trackbars for Hue, Saturation, and Value with broader starting ranges
    cv2.createTrackbar("Lower Hue", "Dynamic Color Filtering", 0, 179, nothing)
    cv2.createTrackbar("Upper Hue", "Dynamic Color Filtering", 179, 179, nothing)
    cv2.createTrackbar("Lower Saturation", "Dynamic Color Filtering", 0, 255, nothing)
    cv2.createTrackbar("Upper Saturation", "Dynamic Color Filtering", 255, 255, nothing)
    cv2.createTrackbar("Lower Value", "Dynamic Color Filtering", 0, 255, nothing)
    cv2.createTrackbar("Upper Value", "Dynamic Color Filtering", 255, 255, nothing)

    while True:
        # Capture the frame from the webcam
        ret, frame = cap.read()

        # If frame was not captured successfully, break the loop
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get current positions of all trackbars
        lower_h = cv2.getTrackbarPos("Lower Hue", "Dynamic Color Filtering")
        upper_h = cv2.getTrackbarPos("Upper Hue", "Dynamic Color Filtering")
        lower_s = cv2.getTrackbarPos("Lower Saturation", "Dynamic Color Filtering")
        upper_s = cv2.getTrackbarPos("Upper Saturation", "Dynamic Color Filtering")
        lower_v = cv2.getTrackbarPos("Lower Value", "Dynamic Color Filtering")
        upper_v = cv2.getTrackbarPos("Upper Value", "Dynamic Color Filtering")

        # Apply color filter based on the trackbar positions
        filtered_frame = filter_color(frame, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v)

        # Show the filtered video frame
        cv2.imshow("Dynamic Color Filtering", filtered_frame)

        # Check if the 'q' key is pressed or the window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # If the window is closed manually, break out of the loop
        if cv2.getWindowProperty("Dynamic Color Filtering", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
