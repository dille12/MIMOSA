
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import time
import json
import os
from datetime import datetime

# Option 1: Simple Console Logger (No blocking)
class ConsoleProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_every=1):
        self.log_every = log_every
        self.best_val_iou = 0
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every == 0:
            current_val_iou = logs.get('val_iou_score', 0)
            current_val_loss = logs.get('val_loss', float('inf'))
            
            # Update best metrics
            if current_val_iou > self.best_val_iou:
                self.best_val_iou = current_val_iou
                iou_indicator = " ⭐"
            else:
                iou_indicator = ""
                
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                loss_indicator = " ⭐"
            else:
                loss_indicator = ""
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1:3d} | LR: {logs.get('learning_rate', 0):.6f}")
            print(f"{'='*60}")
            print(f"Train | Loss: {logs.get('loss', 0):.4f} | IoU: {logs.get('iou_score', 0):.4f}")
            print(f"Val   | Loss: {current_val_loss:.4f}{loss_indicator} | IoU: {current_val_iou:.4f}{iou_indicator}")
            print(f"Best  | Loss: {self.best_val_loss:.4f} | IoU: {self.best_val_iou:.4f}")
            print(f"{'='*60}")

# Option 2: File-based Logger for External Plotting
class FileLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file="training_log.json"):
        self.log_file = log_file
        self.metrics = []
        
    def on_epoch_end(self, epoch, logs=None):
        metric_data = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'iou_score': logs.get('iou_score'),
            'val_iou_score': logs.get('val_iou_score'),
            'learning_rate': logs.get('learning_rate')
        }
        
        self.metrics.append(metric_data)
        
        # Save to file (can be read by external plotting script)
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# Option 3: Non-blocking Matplotlib with Threading
class AsyncPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_every=5):
        self.plot_every = plot_every
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rates': []
        }
        self.plotting = False
        
    def on_epoch_end(self, epoch, logs=None):
        # Store metrics
        self.metrics['epochs'].append(epoch + 1)
        self.metrics['train_loss'].append(logs.get('loss'))
        self.metrics['val_loss'].append(logs.get('val_loss'))
        self.metrics['train_iou'].append(logs.get('iou_score'))
        self.metrics['val_iou'].append(logs.get('val_iou_score'))
        self.metrics['learning_rates'].append(logs.get('learning_rate'))
        
        # Plot asynchronously every N epochs
        if (epoch + 1) % self.plot_every == 0 and not self.plotting:
            self.plotting = True
            plot_thread = Thread(target=self._create_plot)
            plot_thread.daemon = True
            plot_thread.start()
    
    def _create_plot(self):
        try:
            plt.ion()  # Turn on interactive mode
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Progress (Non-blocking)', fontsize=14)
            
            # Plot 1: Loss
            axes[0, 0].clear()
            axes[0, 0].plot(self.metrics['epochs'], self.metrics['train_loss'], 'b-', label='Train')
            axes[0, 0].plot(self.metrics['epochs'], self.metrics['val_loss'], 'r-', label='Val')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: IoU
            axes[0, 1].clear()
            axes[0, 1].plot(self.metrics['epochs'], self.metrics['train_iou'], 'b-', label='Train')
            axes[0, 1].plot(self.metrics['epochs'], self.metrics['val_iou'], 'r-', label='Val')
            axes[0, 1].set_title('IoU Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Learning Rate
            axes[1, 0].clear()
            axes[1, 0].plot(self.metrics['epochs'], self.metrics['learning_rates'], 'g-')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Current stats
            axes[1, 1].clear()
            axes[1, 1].axis('off')
            if self.metrics['epochs']:
                current_epoch = self.metrics['epochs'][-1]
                stats_text = f"""Epoch: {current_epoch}
Train Loss: {self.metrics['train_loss'][-1]:.4f}
Val Loss: {self.metrics['val_loss'][-1]:.4f}
Train IoU: {self.metrics['train_iou'][-1]:.4f}
Val IoU: {self.metrics['val_iou'][-1]:.4f}
LR: {self.metrics['learning_rates'][-1]:.6f}

Best Val IoU: {max(self.metrics['val_iou']):.4f}
Best Val Loss: {min(self.metrics['val_loss']):.4f}"""
                
                axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)  # Brief pause to update display
            
        except Exception as e:
            print(f"Plot error: {e}")
        finally:
            self.plotting = False


import time

# Option 41: Discord
class DiscordProgressCB(tf.keras.callbacks.Callback):
    def __init__(self, log_every=1):
        self.log_every = log_every
        self.startTime = time.time()
        
    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.plot_every == 0:
            sendToDCProgress(epoch, self.startTime-time.time(), logs.get('iou_score'))