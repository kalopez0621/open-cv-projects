import cv2
import numpy as np
import requests
from io import BytesIO
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def extract_patch(texture, patch_size):
    h, w, _ = texture.shape
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    return texture[y:y + patch_size, x:x + patch_size]

def create_texture_from_patches(sample_texture, patch_size, new_width, new_height):
    new_texture = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(0, new_height, patch_size):
        for j in range(0, new_width, patch_size):
            patch_height = min(patch_size, new_height - i)
            patch_width = min(patch_size, new_width - j)
            patch = extract_patch(sample_texture, patch_size)[:patch_height, :patch_width]
            new_texture[i:i + patch_height, j:j + patch_width] = patch
            
    return new_texture

def apply_filters(synthesized_texture):
    # Apply Gaussian Blur
    blurred_texture = cv2.GaussianBlur(synthesized_texture, (5, 5), 0)

    # Sharpening Filter
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened_texture = cv2.filter2D(synthesized_texture, -1, sharpening_kernel)

    return blurred_texture, sharpened_texture

def adjust_brightness(image, factor):
    """Adjust brightness by multiplying by a factor."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_saturation(image, factor):
    """Adjust saturation by converting to HSV and scaling the saturation channel."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)  # Scale saturation
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def adjust_hue(image, offset):
    """Adjust hue by converting to HSV and shifting the hue channel."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + offset) % 180  # Wrap around
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def update_texture():
    url = 'https://storage.googleapis.com/webdesignledger.pub.network/LaT/edd/2017/01/DSC00712-1560x1075.jpg'
    response = requests.get(url)
    image_bytes = BytesIO(response.content)
    image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    sample_texture = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    patch_size = int(patch_size_var.get())
    new_width = int(width_var.get())
    new_height = int(height_var.get())

    synthesized_texture = create_texture_from_patches(sample_texture, patch_size, new_width, new_height)
    
    # Apply filters
    blurred_texture, sharpened_texture = apply_filters(synthesized_texture)

    # Apply color manipulations with slider values
    bright_texture = adjust_brightness(synthesized_texture, brightness_var.get())
    saturated_texture = adjust_saturation(synthesized_texture, saturation_var.get())
    hue_adjusted_texture = adjust_hue(synthesized_texture, 30)  # Example hue offset

    # Convert images to PhotoImage format for Tkinter
    original_image = cv2.cvtColor(sample_texture, cv2.COLOR_BGR2RGB)
    synthesized_image = cv2.cvtColor(synthesized_texture, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.cvtColor(blurred_texture, cv2.COLOR_BGR2RGB)
    sharpened_image = cv2.cvtColor(sharpened_texture, cv2.COLOR_BGR2RGB)
    bright_image = cv2.cvtColor(bright_texture, cv2.COLOR_BGR2RGB)
    saturated_image = cv2.cvtColor(saturated_texture, cv2.COLOR_BGR2RGB)
    hue_image = cv2.cvtColor(hue_adjusted_texture, cv2.COLOR_BGR2RGB)

    # Update the display
    update_display(original_image, synthesized_image, blurred_image, sharpened_image, bright_image, saturated_image, hue_image)

def update_display(original_image, synthesized_image, blurred_image, sharpened_image, bright_image, saturated_image, hue_image):
    # Convert images to PhotoImage
    original_photo = ImageTk.PhotoImage(Image.fromarray(original_image))
    synthesized_photo = ImageTk.PhotoImage(Image.fromarray(synthesized_image))
    blurred_photo = ImageTk.PhotoImage(Image.fromarray(blurred_image))
    sharpened_photo = ImageTk.PhotoImage(Image.fromarray(sharpened_image))
    bright_photo = ImageTk.PhotoImage(Image.fromarray(bright_image))
    saturated_photo = ImageTk.PhotoImage(Image.fromarray(saturated_image))
    hue_photo = ImageTk.PhotoImage(Image.fromarray(hue_image))

    # Update canvas
    original_canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
    original_canvas.image = original_photo  # Keep a reference
    synthesized_canvas.create_image(0, 0, anchor=tk.NW, image=synthesized_photo)
    synthesized_canvas.image = synthesized_photo  # Keep a reference
    blurred_canvas.create_image(0, 0, anchor=tk.NW, image=blurred_photo)
    blurred_canvas.image = blurred_photo  # Keep a reference
    sharpened_canvas.create_image(0, 0, anchor=tk.NW, image=sharpened_photo)
    sharpened_canvas.image = sharpened_photo  # Keep a reference
    bright_canvas.create_image(0, 0, anchor=tk.NW, image=bright_photo)
    bright_canvas.image = bright_photo  # Keep a reference
    saturated_canvas.create_image(0, 0, anchor=tk.NW, image=saturated_photo)
    saturated_canvas.image = saturated_photo  # Keep a reference
    hue_canvas.create_image(0, 0, anchor=tk.NW, image=hue_photo)
    hue_canvas.image = hue_photo  # Keep a reference

# Set up Tkinter GUI
root = tk.Tk()
root.title("Texture Synthesis with Filters")

# Patch Size
patch_size_var = tk.StringVar(value='50')
patch_size_label = ttk.Label(root, text="Patch Size:")
patch_size_label.grid(column=0, row=0, sticky='W')
patch_size_entry = ttk.Entry(root, textvariable=patch_size_var)
patch_size_entry.grid(column=1, row=0, sticky='EW')

# Width
width_var = tk.StringVar(value='640')
width_label = ttk.Label(root, text="Texture Width:")
width_label.grid(column=0, row=1, sticky='W')
width_entry = ttk.Entry(root, textvariable=width_var)
width_entry.grid(column=1, row=1, sticky='EW')

# Height
height_var = tk.StringVar(value='480')
height_label = ttk.Label(root, text="Texture Height:")
height_label.grid(column=0, row=2, sticky='W')
height_entry = ttk.Entry(root, textvariable=height_var)
height_entry.grid(column=1, row=2, sticky='EW')

# Brightness and Saturation sliders
brightness_var = tk.DoubleVar(value=1.0)
saturation_var = tk.DoubleVar(value=1.0)

brightness_label = ttk.Label(root, text="Brightness:")
brightness_label.grid(column=0, row=6, sticky='W')
brightness_slider = ttk.Scale(root, from_=0.0, to=2.0, variable=brightness_var, orient='horizontal')
brightness_slider.grid(column=1, row=6, sticky='EW')

saturation_label = ttk.Label(root, text="Saturation:")
saturation_label.grid(column=0, row=7, sticky='W')
saturation_slider = ttk.Scale(root, from_=0.0, to=3.0, variable=saturation_var, orient='horizontal')
saturation_slider.grid(column=1, row=7, sticky='EW')

# Generate Button
generate_button = ttk.Button(root, text="Generate Texture", command=update_texture)
generate_button.grid(column=0, row=3, columnspan=2, sticky='EW')

# Create canvases for displaying images
canvas_width = 240  # Adjusted width for better fit
canvas_height = 180  # Adjusted height for better fit

original_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
original_canvas.grid(column=0, row=4, sticky='NSEW')

synthesized_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
synthesized_canvas.grid(column=1, row=4, sticky='NSEW')

blurred_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
blurred_canvas.grid(column=0, row=5, sticky='NSEW')

sharpened_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
sharpened_canvas.grid(column=1, row=5, sticky='NSEW')

# Add labels for the canvases
original_label = ttk.Label(root, text="Original Texture")
original_label.grid(column=0, row=6, sticky='EW')

synthesized_label = ttk.Label(root, text="Synthesized Texture")
synthesized_label.grid(column=1, row=6, sticky='EW')

blurred_label = ttk.Label(root, text="Blurred Texture")
blurred_label.grid(column=0, row=7, sticky='EW')

sharpened_label = ttk.Label(root, text="Sharpened Texture")
sharpened_label.grid(column=1, row=7, sticky='EW')

# Create canvases for color manipulated images
bright_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
bright_canvas.grid(column=0, row=8, sticky='NSEW')

saturated_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
saturated_canvas.grid(column=1, row=8, sticky='NSEW')

hue_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
hue_canvas.grid(column=0, row=9, sticky='NSEW')

# Add labels for color manipulated images
bright_label = ttk.Label(root, text="Brightened Texture")
bright_label.grid(column=0, row=10, sticky='EW')

saturated_label = ttk.Label(root, text="Saturated Texture")
saturated_label.grid(column=1, row=10, sticky='EW')

hue_label = ttk.Label(root, text="Hue Adjusted Texture")
hue_label.grid(column=0, row=11, sticky='EW')

# Configure grid to resize properly
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Initial texture generation
update_texture()

# Run the Tkinter main loop
root.mainloop()
