import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Enable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Suppress TensorFlow logging (INFO and WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def prepare_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print(f"Error: Unable to load image at {file_path}. Please check the file path.")
        return None
    img_resized = cv2.resize(img, (224, 224))  # Resize to expected input size
    img_preprocessed = preprocess_input(img_resized)  # Preprocess the image
    img_batch = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension
    return img_batch

def predict_image(file_path):
    img_batch = prepare_image(file_path)
    if img_batch is None:
        return None
    predictions = model.predict(img_batch)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions

# Set the path to the image file
file_path = 'dog.jpg'  # Change this to your image path
predictions = predict_image(file_path)

if predictions:
    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i + 1}: Label: {label}, Score: {score:.2f}")

    # Load and annotate the image
    image = cv2.imread(file_path)

    # Prepare formatted text for the prediction
    formatted_label = label.replace("_", " ").title()  # Convert label to readable format
    formatted_score = f"{score * 100:.0f}"  # Convert score to percentage
    prediction_text = f"{formatted_label}"
    accuracy_text = f"{formatted_score}% accurate"  # Accuracy text

    # Set font and position for the label
    font = cv2.FONT_HERSHEY_SIMPLEX
    position_label = (10, 30)  # Position for the label
    color = (255, 255, 255)  # White color for the text
    thickness = 2
    font_scale = 1

    # Put label on the image
    cv2.putText(image, prediction_text, position_label, font, font_scale, color, thickness, cv2.LINE_AA)

    # Set position for accuracy text (bottom left)
    accuracy_position = (10, image.shape[0] - 10)  # Bottom left corner
    cv2.putText(image, accuracy_text, accuracy_position, font, font_scale, color, thickness, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Image", image)

    while True:
        # Check if the window is still open
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Add a small delay to avoid high CPU usage
        cv2.waitKey(100)

    cv2.destroyAllWindows()
