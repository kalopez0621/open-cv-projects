import cv2
import numpy as np
import requests
from io import BytesIO
import random
import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import csv  # Import CSV for saving feedback

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

    # Convert images to PhotoImage format for Tkinter
    original_image = cv2.cvtColor(sample_texture, cv2.COLOR_BGR2RGB)
    synthesized_image = cv2.cvtColor(synthesized_texture, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.cvtColor(blurred_texture, cv2.COLOR_BGR2RGB)
    sharpened_image = cv2.cvtColor(sharpened_texture, cv2.COLOR_BGR2RGB)

    # Update the display
    update_display(original_image, synthesized_image, blurred_image, sharpened_image)

    # Enable only the submit button
    submit_button.config(state=tk.NORMAL)
    generate_button.config(state=tk.DISABLED)
    close_button.config(state=tk.DISABLED)

def update_display(original_image, synthesized_image, blurred_image, sharpened_image):
    # Convert images to PhotoImage
    original_photo = ImageTk.PhotoImage(Image.fromarray(original_image))
    synthesized_photo = ImageTk.PhotoImage(Image.fromarray(synthesized_image))
    blurred_photo = ImageTk.PhotoImage(Image.fromarray(blurred_image))
    sharpened_photo = ImageTk.PhotoImage(Image.fromarray(sharpened_image))

    # Update canvas
    original_canvas.create_image(0, 0, anchor=tk.NW, image=original_photo)
    original_canvas.image = original_photo  # Keep a reference
    synthesized_canvas.create_image(0, 0, anchor=tk.NW, image=synthesized_photo)
    synthesized_canvas.image = synthesized_photo  # Keep a reference
    blurred_canvas.create_image(0, 0, anchor=tk.NW, image=blurred_photo)
    blurred_canvas.image = blurred_photo  # Keep a reference
    sharpened_canvas.create_image(0, 0, anchor=tk.NW, image=sharpened_photo)
    sharpened_canvas.image = sharpened_photo  # Keep a reference

def check_feedback():
    # Check if all feedback is provided
    if (feedback_vars["synthesized"].get() != "" and
        feedback_vars["blurred"].get() != "" and
        feedback_vars["sharpened"].get() != ""):
        submit_button.config(state=tk.NORMAL)
    else:
        submit_button.config(state=tk.DISABLED)

def submit_feedback():
    feedback_synthesized = feedback_vars["synthesized"].get()
    feedback_blurred = feedback_vars["blurred"].get()
    feedback_sharpened = feedback_vars["sharpened"].get()
    
    # Save feedback to a CSV file
    with open('user_feedback.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([feedback_synthesized, feedback_blurred, feedback_sharpened])
    
    print(f"Synthesized Texture Feedback: {feedback_synthesized}")  # Save or process feedback
    print(f"Blurred Texture Feedback: {feedback_blurred}")
    print(f"Sharpened Texture Feedback: {feedback_sharpened}")

    # Enable buttons to generate new textures or close the window
    generate_button.config(state=tk.NORMAL)
    close_button.config(state=tk.NORMAL)

def reset_feedback():
    for key in feedback_vars:
        feedback_vars[key].set("")  # Reset feedback selections
    submit_button.config(state=tk.DISABLED)  # Disable submit button until new feedback is given

def open_texture_options():
    """Open a dialog to ask the user for texture generation options."""
    options_window = tk.Toplevel(root)
    options_window.title("Texture Generation Options")

    ttk.Label(options_window, text="Choose texture generation option:").pack(pady=10)

    auto_option = tk.BooleanVar(value=True)
    
    ttk.Radiobutton(options_window, text="Generate Automatically", variable=auto_option, value=True).pack(anchor=tk.W)
    ttk.Radiobutton(options_window, text="Make Specific Adjustments", variable=auto_option, value=False).pack(anchor=tk.W)

    def apply_choices():
        if auto_option.get():
            reset_feedback()
            update_texture()  # Automatically generate textures
        else:
            # Create fields for adjusting parameters
            ttk.Label(options_window, text="Patch Size:").pack(anchor=tk.W)
            patch_size_entry = ttk.Entry(options_window)
            patch_size_entry.insert(0, patch_size_var.get())
            patch_size_entry.pack()

            ttk.Label(options_window, text="Texture Width:").pack(anchor=tk.W)
            width_entry = ttk.Entry(options_window)
            width_entry.insert(0, width_var.get())
            width_entry.pack()

            ttk.Label(options_window, text="Texture Height:").pack(anchor=tk.W)
            height_entry = ttk.Entry(options_window)
            height_entry.insert(0, height_var.get())
            height_entry.pack()

            def apply_specific_choices():
                # Get the new values from the entries
                new_patch_size = patch_size_entry.get()
                new_width = width_entry.get()
                new_height = height_entry.get()

                # Update the texture parameters
                patch_size_var.set(new_patch_size)
                width_var.set(new_width)
                height_var.set(new_height)

                reset_feedback()
                update_texture()  # Generate textures with user-defined parameters
                options_window.destroy()

            ttk.Button(options_window, text="Confirm", command=apply_specific_choices).pack(pady=20)

    ttk.Button(options_window, text="Continue", command=apply_choices).pack(pady=20)

def close_window():
    root.destroy()

# Set up Tkinter GUI
root = tk.Tk()
root.title("Texture Synthesis with Filters")

# Patch Size
patch_size_var = tk.StringVar(value='45')  # Default patch size
patch_size_label = ttk.Label(root, text="Patch Size:")
patch_size_label.grid(column=0, row=0, sticky='W')
patch_size_entry = ttk.Entry(root, textvariable=patch_size_var, state='readonly')  # Readonly until new generation
patch_size_entry.grid(column=1, row=0, sticky='EW')

# Width
width_var = tk.StringVar(value='600')  # Default width
width_label = ttk.Label(root, text="Texture Width:")
width_label.grid(column=0, row=1, sticky='W')
width_entry = ttk.Entry(root, textvariable=width_var, state='readonly')  # Readonly until new generation
width_entry.grid(column=1, row=1, sticky='EW')

# Height
height_var = tk.StringVar(value='400')  # Default height
height_label = ttk.Label(root, text="Texture Height:")
height_label.grid(column=0, row=2, sticky='W')
height_entry = ttk.Entry(root, textvariable=height_var, state='readonly')  # Readonly until new generation
height_entry.grid(column=1, row=2, sticky='EW')

# Create canvases for displaying images
canvas_width = 320  # Increased width for better visibility
canvas_height = 240  # Increased height for better visibility

# Create a single row for all four textures
original_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
original_canvas.grid(column=0, row=4, sticky='NSEW')

synthesized_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
synthesized_canvas.grid(column=1, row=4, sticky='NSEW')

blurred_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
blurred_canvas.grid(column=2, row=4, sticky='NSEW')

sharpened_canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
sharpened_canvas.grid(column=3, row=4, sticky='NSEW')

# Create labels for each texture canvas
original_label = ttk.Label(root, text="Original Texture")
original_label.grid(column=0, row=5, sticky='EW')

synthesized_label = ttk.Label(root, text="Synthesized Texture")
synthesized_label.grid(column=1, row=5, sticky='EW')

blurred_label = ttk.Label(root, text="Blurred Texture")
blurred_label.grid(column=2, row=5, sticky='EW')

sharpened_label = ttk.Label(root, text="Sharpened Texture")
sharpened_label.grid(column=3, row=5, sticky='EW')

# Create feedback variables for each texture type
feedback_vars = {
    "synthesized": tk.StringVar(value=""),
    "blurred": tk.StringVar(value=""),
    "sharpened": tk.StringVar(value="")
}

# Feedback section for each generated texture
feedback_label = ttk.Label(root, text="Please provide your feedback:")
feedback_label.grid(column=0, row=6, columnspan=4, sticky='W')

# Feedback for synthesized texture
ttk.Label(root, text="Synthesized Texture:").grid(column=0, row=7, sticky='W')
ttk.Radiobutton(root, text="Happy", variable=feedback_vars["synthesized"], value="Happy", command=check_feedback).grid(column=0, row=8, sticky='W')
ttk.Radiobutton(root, text="Not Happy", variable=feedback_vars["synthesized"], value="Not Happy", command=check_feedback).grid(column=1, row=8, sticky='W')

# Feedback for blurred texture
ttk.Label(root, text="Blurred Texture:").grid(column=0, row=9, sticky='W')
ttk.Radiobutton(root, text="Happy", variable=feedback_vars["blurred"], value="Happy", command=check_feedback).grid(column=0, row=10, sticky='W')
ttk.Radiobutton(root, text="Not Happy", variable=feedback_vars["blurred"], value="Not Happy", command=check_feedback).grid(column=1, row=10, sticky='W')

# Feedback for sharpened texture
ttk.Label(root, text="Sharpened Texture:").grid(column=0, row=11, sticky='W')
ttk.Radiobutton(root, text="Happy", variable=feedback_vars["sharpened"], value="Happy", command=check_feedback).grid(column=0, row=12, sticky='W')
ttk.Radiobutton(root, text="Not Happy", variable=feedback_vars["sharpened"], value="Not Happy", command=check_feedback).grid(column=1, row=12, sticky='W')

# Submit Button for feedback
submit_button = ttk.Button(root, text="Submit Feedback", command=submit_feedback)
submit_button.grid(column=0, row=13, columnspan=4, sticky='EW')
submit_button.config(state=tk.NORMAL)  # Enabled by default when textures are displayed

# Generate New Textures Button (initially disabled)
generate_button = ttk.Button(root, text="Generate New Textures", command=open_texture_options)
generate_button.grid(column=0, row=14, columnspan=2, sticky='EW')  # Adjusted to occupy two columns
generate_button.config(state=tk.DISABLED)  # Initially disabled

# Close Button
close_button = ttk.Button(root, text="Close", command=close_window)
close_button.grid(column=2, row=14, columnspan=2, sticky='EW')  # Adjusted to occupy two columns
close_button.config(state=tk.DISABLED)  # Initially disabled

# Configure grid to resize properly
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_columnconfigure(3, weight=1)

# Initial texture generation
update_texture()

# Run the Tkinter main loop
root.mainloop()

