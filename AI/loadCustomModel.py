import tensorflow as tf
import segmentation_models as sm
from tensorflow import keras

def load_custom_segmentation_model(model_path):
    """
    Load a custom segmentation model with specific loss function handling
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    tf.keras.Model
        Loaded model with custom loss function
    """
    # Define custom objects dictionary
    custom_objects = {
        'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss,
        'iou_score': sm.metrics.iou_score,
        'f1-score': sm.metrics.f1_score
    }
    
    try:
        # Load model with custom objects
        model = keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=True
        )
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback method if direct loading fails
        try:
            # Try loading without compilation
            model = keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=False
            )
            
            # Recompile the model with original loss and metrics
            model.compile(
                optimizer='adam',
                loss=sm.losses.bce_jaccard_loss,
                metrics=[sm.metrics.iou_score, sm.metrics.f1_score]
            )
            return model
        
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise