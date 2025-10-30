import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import glob
from scipy import ndimage
from scipy.ndimage import label
import tensorflow as tf
os.environ['SM_FRAMEWORK'] = 'tf.keras'

from loadCustomModel import load_custom_segmentation_model


class LabelAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Individual Label Particle Annotation Tool with Prediction")
        self.root.geometry("1200x750")
        
        # Configuration
        self.target_size = 128  # Change to 64 if preferred
        self.line_thickness = 2
        
        # State variables
        self.current_label_image = None
        self.original_label_image = None
        self.current_prediction = None
        self.drawing = False
        self.last_x, self.last_y = 0, 0
        self.image_files = []
        self.current_image_index = 0
        self.current_labels = []
        self.current_label_index = 0
        self.input_dir = ""
        self.output_dir = ""
        self.current_image_path = ""
        self.model = None
        self.model_path = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # Directory selection - Row 0
        ttk.Button(control_frame, text="Select Input Directory", 
                  command=self.select_input_dir).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="Select Output Directory", 
                  command=self.select_output_dir).grid(row=0, column=1, padx=(0, 5))
        
        # Model selection - Row 0
        ttk.Button(control_frame, text="Load TensorFlow Model", 
                  command=self.load_model).grid(row=0, column=2, padx=(10, 5))
        
        # Image size selection - Row 0
        ttk.Label(control_frame, text="Target Size:").grid(row=0, column=3, padx=(10, 5))
        self.size_var = tk.StringVar(value="128")
        size_combo = ttk.Combobox(control_frame, textvariable=self.size_var, 
                                 values=["64", "128"], width=5)
        size_combo.grid(row=0, column=4, padx=(0, 5))
        size_combo.bind("<<ComboboxSelected>>", self.on_size_change)
        
        # Start button - Row 0
        ttk.Button(control_frame, text="Start Annotation", 
                  command=self.start_annotation).grid(row=0, column=5, padx=(10, 0))
        
        # Progress info - Row 1
        self.progress_label = ttk.Label(control_frame, text="No images loaded")
        self.progress_label.grid(row=1, column=0, columnspan=3, pady=(5, 0), sticky="w")
        
        # Model status - Row 1
        self.model_status = ttk.Label(control_frame, text="No model loaded", foreground="red")
        self.model_status.grid(row=1, column=3, columnspan=3, pady=(5, 0), sticky="w")
        
        # Label info - Row 2
        self.label_info = ttk.Label(control_frame, text="", foreground="blue")
        self.label_info.grid(row=2, column=0, columnspan=6, pady=(2, 0), sticky="w")
        
        # Canvas frame for images
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        # Original image canvas
        original_frame = ttk.LabelFrame(canvas_frame, text="Original Label", padding="5")
        original_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        
        self.canvas = tk.Canvas(original_frame, width=400, height=400, bg="white")
        self.canvas.grid(row=0, column=0)
        
        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Prediction canvas
        prediction_frame = ttk.LabelFrame(canvas_frame, text="Model Prediction", padding="5")
        prediction_frame.grid(row=0, column=1, sticky="nsew")
        
        self.prediction_canvas = tk.Canvas(prediction_frame, width=400, height=400, bg="white")
        self.prediction_canvas.grid(row=0, column=0)
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=2, sticky="n", padx=(10, 0))
        
        # Control buttons
        ttk.Button(button_frame, text="Reset Label", 
                  command=self.reset_label).grid(row=0, column=0, pady=(0, 5), sticky="ew")
        ttk.Button(button_frame, text="Save & Next Label", 
                  command=self.save_and_next_label).grid(row=1, column=0, pady=(0, 5), sticky="ew")
        ttk.Button(button_frame, text="Skip Label", 
                  command=self.skip_label).grid(row=2, column=0, pady=(0, 5), sticky="ew")
        ttk.Button(button_frame, text="Previous Label", 
                  command=self.previous_label).grid(row=3, column=0, pady=(0, 5), sticky="ew")
        
        # Separator
        ttk.Separator(button_frame, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=10)
        
        # Image navigation
        ttk.Button(button_frame, text="Previous Image", 
                  command=self.previous_image).grid(row=5, column=0, pady=(0, 5), sticky="ew")
        ttk.Button(button_frame, text="Next Image", 
                  command=self.next_image).grid(row=6, column=0, pady=(0, 5), sticky="ew")
        
        # Separator
        ttk.Separator(button_frame, orient="horizontal").grid(row=7, column=0, sticky="ew", pady=10)
        
        # Prediction controls
        ttk.Button(button_frame, text="Generate Prediction", 
                  command=self.generate_prediction).grid(row=8, column=0, pady=(0, 5), sticky="ew")
        ttk.Button(button_frame, text="Use Prediction", 
                  command=self.use_prediction).grid(row=9, column=0, pady=(0, 5), sticky="ew")
        
        # Instructions
        instructions = """
Instructions:
1. Select input/output directories
2. Load TensorFlow model (.h5/.pb/.keras)
3. Click 'Start Annotation' 
4. Each label will be shown individually
5. Generate prediction to see model output
6. Draw black lines to separate particles
7. 'Use Prediction' to copy model output
8. 'Save & Next Label' to continue
9. Navigate between images/labels

The prediction preview helps you see
what the model would output, which you
can use as a starting point or reference
for your manual annotations.
        """
        ttk.Label(button_frame, text=instructions, justify="left", 
                 font=("Arial", 8)).grid(row=10, column=0, pady=(20, 0), sticky="w")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.columnconfigure(1, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
    def load_model(self):
        """Load a TensorFlow model"""
        filetypes = [
            ("TensorFlow Models", "*.h5 *.keras *.pb"),
            ("Keras Models", "*.h5 *.keras"),
            ("TensorFlow SavedModel", "*.pb"),
            ("All files", "*.*")
        ]
        
        model_path = filedialog.askopenfilename(
            title="Select TensorFlow Model",
            filetypes=filetypes
        )
        
        if model_path:
            try:
                # Try loading the model
                if model_path.endswith('.pb'):
                    # For SavedModel format
                    model_dir = os.path.dirname(model_path)
                    self.model = tf.keras.models.load_model(model_dir)
                else:
                    # For .h5 and .keras files
                    self.model = load_custom_segmentation_model(model_path)
                
                self.model_path = model_path
                model_name = os.path.basename(model_path)
                self.model_status.config(text=f"Model loaded: {model_name}", foreground="green")
                
                # Print model summary for debugging
                print("Model loaded successfully!")
                print(f"Input shape: {self.model.input_shape}")
                print(f"Output shape: {self.model.output_shape}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model = None
                self.model_status.config(text="Model loading failed", foreground="red")
        
    def select_input_dir(self):
        self.input_dir = filedialog.askdirectory(title="Select Input Directory")
        if self.input_dir:
            # Find all image files
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
            self.image_files = []
            for ext in extensions:
                self.image_files.extend(glob.glob(os.path.join(self.input_dir, ext)))
                self.image_files.extend(glob.glob(os.path.join(self.input_dir, ext.upper())))
            
            if self.image_files:
                self.progress_label.config(text=f"Found {len(self.image_files)} images")
            else:
                messagebox.showwarning("Warning", "No image files found in selected directory")
                
    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory(title="Select Output Directory")
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def on_size_change(self, event=None):
        self.target_size = int(self.size_var.get())
        if self.current_label_image is not None:
            self.load_current_label()
            
    def start_annotation(self):
        if not self.input_dir or not self.output_dir:
            messagebox.showerror("Error", "Please select both input and output directories")
            return
            
        if not self.image_files:
            messagebox.showerror("Error", "No image files found")
            return
            
        self.current_image_index = 0
        self.load_current_image()
        
    def load_current_image(self):
        if self.current_image_index >= len(self.image_files):
            messagebox.showinfo("Complete", "All images have been processed!")
            return
            
        self.current_image_path = self.image_files[self.current_image_index]
        
        # Load image
        img = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
            return
        
        # Convert to binary if not already
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        labeled_img, num_labels = label(binary_img)
        
        # Extract individual labels
        self.current_labels = []
        for i in range(1, num_labels + 1):  # Skip background (label 0)
            # Create mask for this label
            mask = (labeled_img == i).astype(np.uint8) * 255
            
            # Find bounding box
            coords = np.where(labeled_img == i)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Add some padding
                padding = 5
                y_min = max(0, y_min - padding)
                y_max = min(img.shape[0], y_max + padding + 1)
                x_min = max(0, x_min - padding)
                x_max = min(img.shape[1], x_max + padding + 1)
                
                # Extract the label region
                label_region = mask[y_min:y_max, x_min:x_max]
                
                # Store label info
                self.current_labels.append({
                    'image': label_region,
                    'original': label_region.copy(),
                    'bbox': (x_min, y_min, x_max, y_max),
                    'label_id': i
                })
        
        if not self.current_labels:
            messagebox.showinfo("Info", "No labels found in current image. Moving to next.")
            self.next_image()
            return
            
        # Start with first label
        self.current_label_index = 0
        self.load_current_label()
        
    def load_current_label(self):
        while True:
            if self.current_label_index >= len(self.current_labels):
                # Move to next image
                self.next_image()
                return
                
            # Get current label
            label_data = self.current_labels[self.current_label_index]
            label_img = label_data['image']

            pixels = np.sum(label_img > 0)
            print(f"Label pixels: {pixels}")
            if pixels < 128:
                print("Label is too small, skipping.")
                self.current_label_index += 1
                continue
            
            # Generate output filename
            input_filename = os.path.basename(self.current_image_path)
            name, ext = os.path.splitext(input_filename)
            output_filename = f"{name}_label{label_data['label_id']:03d}_{self.target_size}x{self.target_size}.png"
            if os.path.exists(os.path.join(self.output_dir, output_filename)):
                print("Label already exists, skipping.")
                self.current_label_index += 1
                continue

            break
        
        # Resize to target size
        self.original_label_image = cv2.resize(label_img, (self.target_size, self.target_size))
        self.current_label_image = self.original_label_image.copy()
        
        # Clear prediction
        self.current_prediction = None
        
        # Display images
        self.display_images()
        
        # Auto-generate prediction if model is loaded
        if self.model is not None:
            self.generate_prediction()
        
        # Update progress
        filename = os.path.basename(self.current_image_path)
        self.progress_label.config(
            text=f"Image {self.current_image_index + 1}/{len(self.image_files)}: {filename}"
        )
        self.label_info.config(
            text=f"Label {self.current_label_index + 1}/{len(self.current_labels)} (ID: {label_data['label_id']})"
        )
        
    def display_images(self):
        """Display both original and prediction images"""
        if self.current_label_image is None:
            return
            
        # Display original image
        self.display_image_on_canvas(self.current_label_image, self.canvas)
        
        # Display prediction if available
        if self.current_prediction is not None:
            self.display_image_on_canvas(self.current_prediction, self.prediction_canvas)
        else:
            # Clear prediction canvas
            self.prediction_canvas.delete("all")
            
    def display_image_on_canvas(self, image, canvas):
        """Helper function to display an image on a specific canvas"""
        # Convert to PIL format for display
        pil_image = Image.fromarray(image)
        # Scale up for better visibility (display at 3x size)
        display_size = self.target_size * 3
        pil_image = pil_image.resize((display_size, display_size), Image.NEAREST)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Store reference to prevent garbage collection
        if canvas == self.canvas:
            self.photo = photo
        else:
            self.prediction_photo = photo
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.config(width=display_size, height=display_size)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        
    def generate_prediction(self):
        """Generate prediction using the loaded model"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model loaded. Please load a model first.")
            return
            
        if self.original_label_image is None:
            return
            
        try:
            # Prepare input for model
            # Normalize the image (assuming model expects values between 0 and 1)
            input_image = self.original_label_image.astype(np.float32) / 255.0
            
            # Add batch dimension and possibly channel dimension
            if len(self.model.input_shape) == 4:  # Batch, Height, Width, Channels
                if self.model.input_shape[-1] == 1:  # Single channel
                    input_image = np.expand_dims(input_image, axis=-1)
                elif self.model.input_shape[-1] == 3:  # RGB
                    input_image = np.stack([input_image, input_image, input_image], axis=-1)
            
            # Add batch dimension
            input_batch = np.expand_dims(input_image, axis=0)
            
            print(f"Input shape for model: {input_batch.shape}")
            print(f"Expected input shape: {self.model.input_shape}")
            
            # Generate prediction
            prediction = self.model.predict(input_batch, verbose=0)

            prediction = (prediction > 0.5).astype(np.uint8)

            prediction = input_image * (1 - prediction)  # Apply mask to resized image
            
            # Process prediction output
            if len(prediction.shape) == 4:  # Batch, Height, Width, Channels
                pred_image = prediction[0]  # Remove batch dimension
                if pred_image.shape[-1] == 1:  # Single channel
                    pred_image = pred_image[:, :, 0]
            else:
                pred_image = prediction[0]
            
            # Convert back to 0-255 range
            pred_image = np.clip(pred_image * 255, 0, 255).astype(np.uint8)
            
            # Ensure it's the right size
            if pred_image.shape != (self.target_size, self.target_size):
                pred_image = cv2.resize(pred_image, (self.target_size, self.target_size))
            
            self.current_prediction = pred_image
            
            # Update display
            self.display_images()
            
            print("Prediction generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate prediction: {str(e)}")
            print(f"Prediction error: {str(e)}")
            
    def use_prediction(self):
        """Copy the prediction to the current working image"""
        if self.current_prediction is None:
            messagebox.showwarning("Warning", "No prediction available. Generate a prediction first.")
            return
            
        # Copy prediction to current label image
        self.current_label_image = self.current_prediction.copy()
        
        # Update display
        self.display_images()
        
    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        
    def draw_line(self, event):
        if self.drawing and self.current_label_image is not None:
            # Scale coordinates to image size
            scale = self.target_size / (self.target_size * 3)
            
            x1, y1 = int(self.last_x * scale), int(self.last_y * scale)
            x2, y2 = int(event.x * scale), int(event.y * scale)
            
            # Draw on the actual image (black line = 0)
            cv2.line(self.current_label_image, (x1, y1), (x2, y2), 0, self.line_thickness)
            
            # Update display
            self.display_images()
            
            self.last_x, self.last_y = event.x, event.y
            
    def stop_draw(self, event):
        self.drawing = False
        
    def reset_label(self):
        if self.original_label_image is not None:
            self.current_label_image = self.original_label_image.copy()
            self.display_images()
            
    def save_and_next_label(self):
        if self.current_label_image is None:
            return
            
        # Generate output filename
        input_filename = os.path.basename(self.current_image_path)
        name, ext = os.path.splitext(input_filename)
        label_data = self.current_labels[self.current_label_index]
        
        output_filename = f"{name}_label{label_data['label_id']:03d}_{self.target_size}x{self.target_size}.png"
        valoutput_filename = f"VAL_{name}_label{label_data['label_id']:03d}_{self.target_size}x{self.target_size}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        valoutput_path = os.path.join(self.output_dir, valoutput_filename)

        VAL = self.current_labels[self.current_label_index]["original"]
        VAL = cv2.resize(VAL, (self.target_size, self.target_size))
        TRAIN = VAL - self.current_label_image.copy()
        
        # Save the annotated label
        cv2.imwrite(output_path, TRAIN)
        cv2.imwrite(valoutput_path, VAL)
        
        # Move to next label
        self.current_label_index += 1
        self.load_current_label()
        
    def skip_label(self):
        self.current_label_index += 1
        self.load_current_label()
        
    def previous_label(self):
        if self.current_label_index > 0:
            self.current_label_index -= 1
            self.load_current_label()
            
    def next_image(self):
        self.current_image_index += 1
        self.load_current_image()
        
    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

def main():
    root = tk.Tk()
    app = LabelAnnotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()