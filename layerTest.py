import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from tensorflow.keras.models import load_model
from AI.loadCustomModel import load_custom_segmentation_model

model = load_custom_segmentation_model("C:/Users/cgvisa/Documents/VSCode/NEURAL NETWORKS/ITER16_RETRAIN_4166IMAGES_SHAPE_256x256x1_1.keras")

# Method 1: Basic layer information with shapes
print("=== Method 1: Basic Layer Info ===")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}")
    print(f"  Type: {type(layer).__name__}")
    print(f"  Built: {layer.built}")
    
    # Input shape
    if hasattr(layer, 'input_shape') and layer.input_shape is not None:
        print(f"  Input shape: {layer.input_shape}")
    
    # Output shape
    if hasattr(layer, 'output_shape') and layer.output_shape is not None:
        print(f"  Output shape: {layer.output_shape}")
    
    print("-" * 40)

# Method 2: Using model.summary() - most comprehensive
print("\n=== Method 2: Model Summary ===")
model.summary()

# Method 3: More detailed layer analysis
print("\n=== Method 3: Detailed Layer Analysis ===")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({type(layer).__name__})")
    
    # Try to get input/output shapes in different ways
    try:
        if hasattr(layer, 'input_spec') and layer.input_spec:
            print(f"  Input spec: {layer.input_spec}")
        
        if hasattr(layer, 'output_shape'):
            print(f"  Output shape: {layer.output_shape}")
            
        # For layers with weights
        if layer.weights:
            print(f"  Trainable params: {layer.count_params()}")
            for j, weight in enumerate(layer.weights):
                print(f"    Weight {j}: {weight.name} - Shape: {weight.shape}")
    
    except Exception as e:
        print(f"  Error getting details: {e}")
    
    print("-" * 50)

# Method 4: Compact view of all layer shapes
print("\n=== Method 4: Compact Shape Overview ===")
for i, layer in enumerate(model.layers):
    input_shape = getattr(layer, 'input_shape', 'N/A')
    output_shape = getattr(layer, 'output_shape', 'N/A')
    print(f"{i:2d}. {layer.name:<20} | In: {str(input_shape):<20} | Out: {str(output_shape)}")

# Method 5: Just the essential info
print("\n=== Method 5: Essential Info Only ===")
total_params = model.count_params()
print(f"Model: {model.name if hasattr(model, 'name') else 'Unnamed'}")
print(f"Total parameters: {total_params:,}")
print(f"Total layers: {len(model.layers)}")

for i, layer in enumerate(model.layers):
    layer_type = type(layer).__name__
    output_shape = getattr(layer, 'output_shape', 'Unknown')
    params = layer.count_params() if hasattr(layer, 'count_params') else 0
    print(f"{i:2d}. {layer.name:<25} [{layer_type:<15}] -> {str(output_shape):<25} ({params:,} params)")