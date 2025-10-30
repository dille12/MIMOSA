import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys
from datetime import datetime
import argparse
class TrainingMonitor:
    def __init__(self, log_file="logs/training_metrics.json", update_interval=2000):
        """
        Real-time training monitor that reads from JSON log file
        
        Args:
            log_file (str): Path to the JSON log file
            update_interval (int): Update interval in milliseconds
        """
        self.log_file = log_file
        self.update_interval = update_interval
        self.last_epoch = 0
        self.data = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rates': [],
            'elapsed_times': []
        }
        
        # Set up the plot
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16)
        
        # Initialize plots
        self.setup_plots()
        
    def setup_plots(self):
        """Initialize all subplot configurations"""
        # Loss plot
        self.axes[0, 0].set_title('Loss Over Time')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # IoU plot
        self.axes[0, 1].set_title('IoU Score Over Time')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('IoU Score')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        self.axes[0, 2].set_title('Learning Rate Over Time')
        self.axes[0, 2].set_xlabel('Epoch')
        self.axes[0, 2].set_ylabel('Learning Rate')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        # Training time plot
        self.axes[1, 0].set_title('Training Time')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Elapsed Time (minutes)')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Stats panel
        self.axes[1, 1].set_title('Current Statistics')
        self.axes[1, 1].axis('off')
        
        # Progress panel
        self.axes[1, 2].set_title('Progress Summary')
        self.axes[1, 2].axis('off')
        
    def read_log_file(self):
        """Read and parse the JSON log file"""
        try:
            if not os.path.exists(self.log_file):
                return False
                
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
                
            if 'metrics' not in log_data or not log_data['metrics']:
                return False
                
            metrics = log_data['metrics']
            current_epoch = len(metrics)
            
            # Only update if we have new data
            if current_epoch <= self.last_epoch:
                return False
                
            self.last_epoch = current_epoch
            
            # Extract data
            self.data['epochs'] = [m['epoch'] for m in metrics]
            self.data['train_loss'] = [m['loss'] for m in metrics]
            self.data['val_loss'] = [m['val_loss'] for m in metrics]
            self.data['train_iou'] = [m['iou_score'] for m in metrics]
            self.data['val_iou'] = [m['val_iou_score'] for m in metrics]
            self.data['learning_rates'] = [m['learning_rate'] for m in metrics]
            self.data['elapsed_times'] = [m['elapsed_seconds'] / 60 for m in metrics]  # Convert to minutes
            
            return True
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error reading log file: {e}")
            return False
    
    def update_plots(self, frame):
        """Update all plots with new data"""
        if not self.read_log_file():
            return
            
        # Clear all plots
        for ax in self.axes.flat:
            if ax.get_title() != 'Current Statistics' and ax.get_title() != 'Progress Summary':
                ax.clear()
        
        epochs = self.data['epochs']
        if not epochs:
            return
            
        # Plot 1: Loss
        self.axes[0, 0].plot(epochs, self.data['train_loss'], 'b-', label='Training Loss', linewidth=2)
        self.axes[0, 0].plot(epochs, self.data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        self.axes[0, 0].set_title('Loss Over Time')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: IoU
        self.axes[0, 1].plot(epochs, self.data['train_iou'], 'b-', label='Training IoU', linewidth=2)
        self.axes[0, 1].plot(epochs, self.data['val_iou'], 'r-', label='Validation IoU', linewidth=2)
        self.axes[0, 1].set_title('IoU Score Over Time')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('IoU Score')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate
        self.axes[0, 2].plot(epochs, self.data['learning_rates'], 'g-', linewidth=2)
        self.axes[0, 2].set_title('Learning Rate Over Time')
        self.axes[0, 2].set_xlabel('Epoch')
        self.axes[0, 2].set_ylabel('Learning Rate')
        self.axes[0, 2].set_yscale('log')
        self.axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Training Time
        self.axes[1, 0].plot(epochs, self.data['elapsed_times'], 'purple', linewidth=2)
        self.axes[1, 0].set_title('Training Time')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Elapsed Time (minutes)')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Stats panel
        self.axes[1, 1].clear()
        self.axes[1, 1].axis('off')
        
        current_epoch = epochs[-1]
        current_stats = f"""Current Epoch: {current_epoch}
        
Training Loss: {self.data['train_loss'][-1]:.4f}
Validation Loss: {self.data['val_loss'][-1]:.4f}

Training IoU: {self.data['train_iou'][-1]:.4f}
Validation IoU: {self.data['val_iou'][-1]:.4f}

Learning Rate: {self.data['learning_rates'][-1]:.6f}
Elapsed Time: {self.data['elapsed_times'][-1]:.1f} min

Best Val IoU: {max(self.data['val_iou']):.4f}
Best Val Loss: {min(self.data['val_loss']):.4f}"""
        
        self.axes[1, 1].text(0.05, 0.95, current_stats, transform=self.axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        # Progress panel
        self.axes[1, 2].clear()
        self.axes[1, 2].axis('off')
        
        # Calculate some progress metrics
        recent_epochs = min(10, len(epochs))
        if recent_epochs > 1:
            recent_val_loss = self.data['val_loss'][-recent_epochs:]
            recent_val_iou = self.data['val_iou'][-recent_epochs:]
            
            loss_trend = "↓" if recent_val_loss[-1] < recent_val_loss[0] else "↑"
            iou_trend = "↑" if recent_val_iou[-1] > recent_val_iou[0] else "↓"
            
            avg_time_per_epoch = self.data['elapsed_times'][-1] / current_epoch
            
            progress_text = f"""Progress Summary:
            
Epochs Completed: {current_epoch}
Avg Time/Epoch: {avg_time_per_epoch:.1f} min

Recent Trends (last {recent_epochs} epochs):
Val Loss: {loss_trend} ({recent_val_loss[-1]:.4f})
Val IoU: {iou_trend} ({recent_val_iou[-1]:.4f})

File: {os.path.basename(self.log_file)}
Last Update: {datetime.now().strftime('%H:%M:%S')}"""
            
            self.axes[1, 2].text(0.05, 0.95, progress_text, transform=self.axes[1, 2].transAxes,
                               fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """Start the real-time monitoring"""
        print(f"Starting real-time monitoring of: {self.log_file}")
        print(f"Update interval: {self.update_interval/1000:.1f} seconds")
        print("Close the plot window to stop monitoring.")
        
        # Set up animation
        ani = animation.FuncAnimation(self.fig, self.update_plots, 
                                    interval=self.update_interval, 
                                    blit=False, cache_frame_data=False)
        
        plt.show()
import time
if __name__ == "__main__":
    log_file = "training_log.json"
    update_interval = 5000
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        print("Make sure your training script is running and generating the log file.")
        sys.exit(1)
    
    # Start monitoring
    monitor = TrainingMonitor(log_file, update_interval)
    time.sleep(2)
    monitor.start_monitoring()