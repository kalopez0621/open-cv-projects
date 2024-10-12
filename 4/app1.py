import cv2
import numpy as np
import requests
from io import BytesIO
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Import PIL for image handling

# Your existing texture synthesis functions
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

def update_texture():
    # Fetch image and create texture
    url = 'https://storage.googleapis.com/webdesignledger.pub.network/LaT/edd/2017/01/DSC00712-1560x1075.jpg'
    response = requests.get(url)
    image_bytes = BytesIO(response.content)
    image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    sample_texture = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    patch_size = int(patch_size_var.get())
    new_width = int(width_var.get())
    new_height = int(height_var.get())

    synthesized_texture = create_texture_from_patches(sample_texture, patch_size, new_width, new_height)

    # Convert images to PhotoImage format for Tkinter
    original_image = cv2.cvtColor(sample_texture, cv2.COLOR_BGR2RGB)
    synthesized_image = cv2.cvtColor(synthesized_texture, cv2.COLOR_BGR2RGB)

    # Update the display
    update_display(original_image, synthesized_image)

def update_display(original_image, synthesized_image):
    # Resize images for display
    original_image = Image.fromarray(original_image)
    synthesized_image = Image.fromarray(synthesized_image)

    # Convert to PhotoImage
    original_photo = ImageTk.PhotoImage(original_image)
    synthesized_photo = ImageTk.PhotoImage(synthesized_image)

    # Update canvas
    original_canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
    original_canvas.image = original_photo  # Keep a reference
    synthesized_canvas.create_image(0, 0, anchor=tk.NW, image=synthesized_photo)
    synthesized_canvas.image = synthesized_photo  # Keep a reference

# Set up Tkinter GUI
root = tk.Tk()
root.title("Texture Synthesis")

# Patch Size
patch_size_var = tk.StringVar(value='50')
patch_size_label = ttk.Label(root, text="Patch Size:")
patch_size_label.grid(column=0, row=0)
patch_size_entry = ttk.Entry(root, textvariable=patch_size_var)
patch_size_entry.grid(column=1, row=0)

# Width
width_var = tk.StringVar(value='640')
width_label = ttk.Label(root, text="Texture Width:")
width_label.grid(column=0, row=1)
width_entry = ttk.Entry(root, textvariable=width_var)
width_entry.grid(column=1, row=1)

# Height
height_var = tk.StringVar(value='480')
height_label = ttk.Label(root, text="Texture Height:")
height_label.grid(column=0, row=2)
height_entry = ttk.Entry(root, textvariable=height_var)
height_entry.grid(column=1, row=2)

# Generate Button
generate_button = ttk.Button(root, text="Generate Texture", command=update_texture)
generate_button.grid(column=0, row=3, columnspan=2)

# Create canvas for displaying images
original_canvas = tk.Canvas(root, width=320, height=240)
original_canvas.grid(column=0, row=4)

synthesized_canvas = tk.Canvas(root, width=320, height=240)
synthesized_canvas.grid(column=1, row=4)

# Initial texture generation
update_texture()

root.mainloop()
