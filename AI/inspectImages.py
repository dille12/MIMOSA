import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import glob

class ImagePairViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Pair Viewer")
        self.root.geometry("1200x800")
        
        # Image lists and current index
        self.train_images = []
        self.validate_images = []
        self.current_index = 0
        
        # Load images
        self.load_images()
        
        # Create GUI
        self.create_widgets()
        
        # Display first pair
        if self.train_images and self.validate_images:
            self.display_current_pair()
        else:
            messagebox.showwarning("No Images", "No matching image pairs found in train_images and validate_images folders")
    
    def load_images(self):
        """Load image file paths from both folders"""
        train_folder = "AI/train_images"
        validate_folder = "AI/validate_images"
        
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
        
        # Get all image files from both folders
        train_files = []
        validate_files = []
        
        if os.path.exists(train_folder):
            for ext in extensions:
                train_files.extend(glob.glob(os.path.join(train_folder, ext)))
                train_files.extend(glob.glob(os.path.join(train_folder, ext.upper())))
        
        if os.path.exists(validate_folder):
            for ext in extensions:
                validate_files.extend(glob.glob(os.path.join(validate_folder, ext)))
                validate_files.extend(glob.glob(os.path.join(validate_folder, ext.upper())))
        
        # Sort files to ensure consistent pairing
        train_files.sort()
        validate_files.sort()
        
        # Match pairs by filename (without path)
        train_dict = {os.path.basename(f): f for f in train_files}
        validate_dict = {os.path.basename(f): f for f in validate_files}
        
        # Find matching pairs
        common_names = set(train_dict.keys()) & set(validate_dict.keys())
        
        for name in sorted(common_names):
            self.train_images.append(train_dict[name])
            self.validate_images.append(validate_dict[name])
        
        print(f"Found {len(self.train_images)} matching image pairs")
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title labels
        ttk.Label(main_frame, text="Training Image", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 10))
        ttk.Label(main_frame, text="Validation Image", font=('Arial', 12, 'bold')).grid(row=0, column=1, pady=(0, 10))
        
        # Image frames
        self.train_frame = ttk.Frame(main_frame, relief='sunken', borderwidth=2)
        self.train_frame.grid(row=1, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.validate_frame = ttk.Frame(main_frame, relief='sunken', borderwidth=2)
        self.validate_frame.grid(row=1, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image labels
        self.train_label = ttk.Label(self.train_frame)
        self.train_label.pack(expand=True)
        
        self.validate_label = ttk.Label(self.validate_frame)
        self.validate_label.pack(expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Navigation buttons
        ttk.Button(control_frame, text="Previous", command=self.prev_pair).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Next", command=self.next_pair).pack(side=tk.LEFT, padx=(0, 10))
        
        # Index label
        self.index_label = ttk.Label(control_frame, text="", font=('Arial', 10))
        self.index_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Filename labels
        self.filename_frame = ttk.Frame(main_frame)
        self.filename_frame.grid(row=3, column=0, columnspan=2, pady=(5, 0))
        
        self.train_filename_label = ttk.Label(self.filename_frame, text="", font=('Arial', 9), foreground='blue')
        self.train_filename_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.validate_filename_label = ttk.Label(self.filename_frame, text="", font=('Arial', 9), foreground='blue')
        self.validate_filename_label.pack(side=tk.LEFT)
    
    def load_and_resize_image(self, image_path, max_width=500, max_height=400):
        """Load and resize an image to fit within the specified dimensions"""
        try:
            image = Image.open(image_path)
            
            # Calculate resize ratio
            width_ratio = max_width / image.width
            height_ratio = max_height / image.height
            ratio = min(width_ratio, height_ratio)
            
            # Resize if necessary
            if ratio < 1:
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def display_current_pair(self):
        """Display the current pair of images"""
        if not self.train_images or not self.validate_images:
            return
        
        # Load and display train image
        train_photo = self.load_and_resize_image(self.train_images[self.current_index])
        if train_photo:
            self.train_label.configure(image=train_photo)
            self.train_label.image = train_photo  # Keep a reference
        
        # Load and display validate image
        validate_photo = self.load_and_resize_image(self.validate_images[self.current_index])
        if validate_photo:
            self.validate_label.configure(image=validate_photo)
            self.validate_label.image = validate_photo  # Keep a reference
        
        # Update index label
        self.index_label.configure(text=f"{self.current_index + 1} / {len(self.train_images)}")
        
        # Update filename labels
        self.train_filename_label.configure(text=os.path.basename(self.train_images[self.current_index]))
        self.validate_filename_label.configure(text=os.path.basename(self.validate_images[self.current_index]))
    
    def next_pair(self):
        """Show next image pair"""
        if self.current_index < len(self.train_images) - 1:
            self.current_index += 1
            self.display_current_pair()
    
    def prev_pair(self):
        """Show previous image pair"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_pair()

def main():
    root = tk.Tk()
    app = ImagePairViewer(root)
    
    # Keyboard bindings
    root.bind('<Left>', lambda e: app.prev_pair())
    root.bind('<Right>', lambda e: app.next_pair())
    root.bind('<space>', lambda e: app.next_pair())
    
    root.mainloop()

if __name__ == "__main__":
    main()