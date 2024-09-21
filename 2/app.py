

import cv2
import numpy as np

def filter_color(image, lower_bound, upper_bound):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the specific color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Use the mask to filter the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

def main():
    # Load an image from your local directory
    image = cv2.imread('Emely3.jpg')

    # Define color range for filtering (HSV format)
    # Example: Blue color range
    lower_bound = np.array([100, 150, 0])
    upper_bound = np.array([140, 255, 255])

    # Apply color filtering
    filtered_image = filter_color(image, lower_bound, upper_bound)

    # Display the filtered image
    cv2.imshow("Filtered Image", filtered_image)

    # Wait until a key is pressed and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
