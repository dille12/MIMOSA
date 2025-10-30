
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Import your original modules (you'll need to ensure these are available)
from ai_core import sliding_window, genSobel, genContrastEdges, genContrastEdges2
from MODELS import unet1, unet2, unet3, unet_upscale, unet_segment
from loadCustomModel import load_custom_segmentation_model
import sendtodc
import benchmarker.runtests

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TensorFlow Model Training Interface")
        self.root.geometry("1200x800")
        
        # Training state
        self.training_thread = None
        self.is_training = False
        self.history = None
        
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_config_tab()
        self.create_data_tab()
        self.create_training_tab()
        self.create_results_tab()
        
        # Redirect stdout to capture print statements
        self.setup_output_capture()
        
    def create_config_tab(self):
        """Create the configuration tab"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")
        
        # Create scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Model Configuration
        model_frame = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        # Input Shape
        ttk.Label(model_frame, text="Input Shape:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.input_shape_var = tk.IntVar(value=128)
        ttk.Entry(model_frame, textvariable=self.input_shape_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Stride
        ttk.Label(model_frame, text="Stride:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.stride_var = tk.IntVar(value=64)
        ttk.Entry(model_frame, textvariable=self.stride_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Input Depth
        ttk.Label(model_frame, text="Input Depth:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.input_depth_var = tk.IntVar(value=1)
        ttk.Entry(model_frame, textvariable=self.input_depth_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # Iteration
        ttk.Label(model_frame, text="Iteration:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.iteration_var = tk.IntVar(value=22)
        ttk.Entry(model_frame, textvariable=self.iteration_var, width=10).grid(row=3, column=1, sticky='w', padx=5, pady=2)
        
        # Comment
        ttk.Label(model_frame, text="Comment:").grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.comment_var = tk.StringVar(value="SHALLOW")
        ttk.Entry(model_frame, textvariable=self.comment_var, width=20).grid(row=4, column=1, sticky='w', padx=5, pady=2)
        
        # Model Selection
        ttk.Label(model_frame, text="Model Version:").grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.model_var = tk.StringVar(value="unet_segment")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=["unet1", "unet2", "unet3", "unet_upscale", "unet_segment"])
        model_combo.grid(row=5, column=1, sticky='w', padx=5, pady=2)
        
        # Training Configuration
        train_frame = ttk.LabelFrame(scrollable_frame, text="Training Configuration", padding=10)
        train_frame.pack(fill='x', padx=5, pady=5)
        
        # Batch Size
        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Entry(train_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Epochs
        ttk.Label(train_frame, text="Max Epochs:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.epochs_var = tk.IntVar(value=500)
        ttk.Entry(train_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Learning Rate
        ttk.Label(train_frame, text="Learning Rate:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(train_frame, textvariable=self.lr_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # Options Configuration
        options_frame = ttk.LabelFrame(scrollable_frame, text="Options", padding=10)
        options_frame.pack(fill='x', padx=5, pady=5)
        
        # Checkboxes
        self.use_sm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use Segmentation Models", variable=self.use_sm_var).pack(anchor='w')
        
        self.rgb_data_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="RGB Data", variable=self.rgb_data_var).pack(anchor='w')
        
        self.augment_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Data Augmentation", variable=self.augment_var).pack(anchor='w')
        
        self.super_resolution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Super Resolution", variable=self.super_resolution_var).pack(anchor='w')
        
        self.particle_separation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Particle Separation", variable=self.particle_separation_var).pack(anchor='w')
        
        self.skip_train_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Skip Training", variable=self.skip_train_var).pack(anchor='w')
        
        self.send_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Send Data", variable=self.send_data_var).pack(anchor='w')
        
        # File Paths
        paths_frame = ttk.LabelFrame(scrollable_frame, text="File Paths", padding=10)
        paths_frame.pack(fill='x', padx=5, pady=5)
        
        # Retrain Model Path
        ttk.Label(paths_frame, text="Retrain Model:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.retrain_model_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.retrain_model_var, width=40).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_file(self.retrain_model_var, "Select Model File")).grid(row=0, column=2, padx=5)
        
        # Image Directory
        ttk.Label(paths_frame, text="Image Directory:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.image_dir_var = tk.StringVar(value="AI/train_images/")
        ttk.Entry(paths_frame, textvariable=self.image_dir_var, width=40).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.image_dir_var)).grid(row=1, column=2, padx=5)
        
        # Mask Directory
        ttk.Label(paths_frame, text="Mask Directory:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.mask_dir_var = tk.StringVar(value="AI/validate_images/")
        ttk.Entry(paths_frame, textvariable=self.mask_dir_var, width=40).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.mask_dir_var)).grid(row=2, column=2, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_data_tab(self):
        """Create the data loading tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Loading")
        
        # Data loading controls
        control_frame = ttk.LabelFrame(data_frame, text="Data Loading Controls", padding=10)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load Data", command=self.load_data).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Visualize Data", command=self.visualize_data).pack(side='left', padx=5)
        
        # Data info
        self.data_info = scrolledtext.ScrolledText(data_frame, height=10)
        self.data_info.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_training_tab(self):
        """Create the training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training")
        
        # Training controls
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Training", 
                                      command=self.start_training)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training", 
                                     command=self.stop_training, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100, length=200)
        self.progress_bar.pack(side='left', padx=10)
        
        # Training log
        log_frame = ttk.LabelFrame(training_frame, text="Training Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=20)
        self.training_log.pack(fill='both', expand=True)
        
    def create_results_tab(self):
        """Create the results tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results controls
        control_frame = ttk.Frame(results_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="Plot Training History", 
                  command=self.plot_training_history).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Model", 
                  command=self.save_model).pack(side='left', padx=5)
        
        # Plot area
        self.plot_frame = ttk.Frame(results_frame)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
    def browse_file(self, var, title):
        """Browse for a file"""
        filename = filedialog.askopenfilename(title=title)
        if filename:
            var.set(filename)
            
    def browse_directory(self, var):
        """Browse for a directory"""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
            
    def setup_output_capture(self):
        """Setup output capture to redirect print statements to GUI"""
        self.old_stdout = sys.stdout
        sys.stdout = StringIO()
        
    def update_log(self, message):
        """Update the training log"""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
        self.root.update_idletasks()
        
    def load_data(self):
        """Load training data"""
        try:
            self.update_log("Loading data...")
            # Here you would call your data loading functions
            # This is a placeholder - you'll need to integrate your actual data loading code
            self.data_info.delete(1.0, tk.END)
            self.data_info.insert(tk.END, f"Data loading configuration:\n")
            self.data_info.insert(tk.END, f"Input Shape: {self.input_shape_var.get()}\n")
            self.data_info.insert(tk.END, f"Stride: {self.stride_var.get()}\n")
            self.data_info.insert(tk.END, f"RGB Data: {self.rgb_data_var.get()}\n")
            self.data_info.insert(tk.END, f"Augmentation: {self.augment_var.get()}\n")
            self.data_info.insert(tk.END, f"Image Directory: {self.image_dir_var.get()}\n")
            self.data_info.insert(tk.END, f"Mask Directory: {self.mask_dir_var.get()}\n")
            
            self.update_log("Data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def visualize_data(self):
        """Visualize loaded data"""
        try:
            self.update_log("Visualizing data...")
            # Placeholder for data visualization
            messagebox.showinfo("Info", "Data visualization would be displayed here")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize data: {str(e)}")
            
    def start_training(self):
        """Start the training process"""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
            
        # Validate inputs
        if not self.validate_inputs():
            return
            
        self.is_training = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            if self.input_shape_var.get() <= 0:
                raise ValueError("Input shape must be positive")
            if self.stride_var.get() <= 0:
                raise ValueError("Stride must be positive")
            if self.epochs_var.get() <= 0:
                raise ValueError("Epochs must be positive")
            if self.batch_size_var.get() <= 0:
                raise ValueError("Batch size must be positive")
            return True
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e))
            return False
            
    def training_worker(self):
        """Worker function that runs the training"""
        try:
            self.update_log("Starting training...")
            self.update_log(f"Configuration:")
            self.update_log(f"  Input Shape: {self.input_shape_var.get()}")
            self.update_log(f"  Stride: {self.stride_var.get()}")
            self.update_log(f"  Batch Size: {self.batch_size_var.get()}")
            self.update_log(f"  Max Epochs: {self.epochs_var.get()}")
            self.update_log(f"  Learning Rate: {self.lr_var.get()}")
            self.update_log(f"  Model: {self.model_var.get()}")
            
            # Here you would integrate your actual training code
            # For now, this is a simulation
            for epoch in range(1, min(self.epochs_var.get() + 1, 11)):  # Simulate 10 epochs max
                if not self.is_training:  # Check if training was stopped
                    break
                    
                self.update_log(f"Epoch {epoch}/{self.epochs_var.get()}")
                
                # Simulate training progress
                import time
                time.sleep(1)  # Simulate training time
                
                # Update progress bar
                progress = (epoch / min(self.epochs_var.get(), 10)) * 100
                self.progress_var.set(progress)
                
                # Simulate loss values
                train_loss = 1.0 - (epoch * 0.1) + np.random.normal(0, 0.05)
                val_loss = 1.0 - (epoch * 0.08) + np.random.normal(0, 0.1)
                
                self.update_log(f"  Training Loss: {train_loss:.4f}")
                self.update_log(f"  Validation Loss: {val_loss:.4f}")
                
            if self.is_training:
                self.update_log("Training completed successfully!")
                messagebox.showinfo("Success", "Training completed successfully!")
            else:
                self.update_log("Training stopped by user.")
                
        except Exception as e:
            self.update_log(f"Training failed: {str(e)}")
            messagebox.showerror("Training Error", f"Training failed: {str(e)}")
        finally:
            self.is_training = False
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.progress_var.set(0)
            
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.update_log("Stopping training...")
        
    def plot_training_history(self):
        """Plot training history"""
        try:
            # Clear previous plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
                
            # Create a simple plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Simulate some training history data
            epochs = range(1, 11)
            train_loss = [1.0 - i*0.1 + np.random.normal(0, 0.05) for i in epochs]
            val_loss = [1.0 - i*0.08 + np.random.normal(0, 0.1) for i in epochs]
            train_acc = [0.5 + i*0.05 + np.random.normal(0, 0.02) for i in epochs]
            val_acc = [0.5 + i*0.04 + np.random.normal(0, 0.03) for i in epochs]
            
            ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
            ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
            ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            
            # Embed plot in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to plot training history: {str(e)}")
            
    def save_model(self):
        """Save the trained model"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".keras",
                filetypes=[("Keras files", "*.keras"), ("All files", "*.*")]
            )
            if filename:
                # Here you would save your actual model
                self.update_log(f"Model would be saved to: {filename}")
                messagebox.showinfo("Success", f"Model saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")

def main():
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()