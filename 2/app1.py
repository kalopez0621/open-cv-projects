import cv2
import numpy as np

# Function to filter the image based on current slider values and replace with a specific color
def filter_color(image, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v, color):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the specific color range
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create the mask
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Create an empty image with the same shape as the original, filled with the specific color
    color_image = np.zeros_like(image)
    color_image[:, :] = color  # Fill the whole image with the specific color

    # Combine the color_image with the original using the mask (show the specific color where mask is valid)
    result = cv2.bitwise_and(color_image, color_image, mask=mask)

    return result

# Callback function for the trackbars (does nothing, but required by createTrackbar)
def nothing(x):
    pass

def main():
    # Load an image from your local directory
    image = cv2.imread('Emely3.jpg')  # Replace with your image path
    
    # Check if the image was loaded correctly
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Resize image if it's too large for display
    image = cv2.resize(image, (640, 480))

    # Create windows for combined image display
    cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)

    # Create trackbars for Hue, Saturation, and Value
    cv2.createTrackbar("Lower Hue", "Combined Image", 50, 179, nothing)
    cv2.createTrackbar("Upper Hue", "Combined Image", 100, 179, nothing)
    cv2.createTrackbar("Lower Saturation", "Combined Image", 50, 255, nothing)
    cv2.createTrackbar("Upper Saturation", "Combined Image", 255, 255, nothing)
    cv2.createTrackbar("Lower Value", "Combined Image", 50, 255, nothing)
    cv2.createTrackbar("Upper Value", "Combined Image", 255, 255, nothing)

    # Choose the specific color you want to apply to the filtered areas (BGR format)
    specific_color = [0, 255, 0]  # Example: Green (BGR format)

    while True:
        # Get current positions of all trackbars
        lower_h = cv2.getTrackbarPos("Lower Hue", "Combined Image")
        upper_h = cv2.getTrackbarPos("Upper Hue", "Combined Image")
        lower_s = cv2.getTrackbarPos("Lower Saturation", "Combined Image")
        upper_s = cv2.getTrackbarPos("Upper Saturation", "Combined Image")
        lower_v = cv2.getTrackbarPos("Lower Value", "Combined Image")
        upper_v = cv2.getTrackbarPos("Upper Value", "Combined Image")

        # Apply color filter based on the trackbar positions and replace with a specific color
        filtered_image = filter_color(image, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v, specific_color)

        # Combine the original and filtered images side by side
        combined_image = np.hstack((image, filtered_image))

        # Show the combined image
        cv2.imshow("Combined Image", combined_image)

        # Check if the 'q' key is pressed or the window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # If the window is closed manually, break out of the loop
        if cv2.getWindowProperty("Combined Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Destroy all windows when finished
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
