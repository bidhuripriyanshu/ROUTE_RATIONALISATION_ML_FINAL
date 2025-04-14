import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

def visualize_model(model_path='best_model.h5'):
    # Load the trained model
    model = load_model(model_path)
    
    # Create a directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Plot model architecture
    plot_model(
        model,
        to_file='visualizations/model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )
    
    # Print model summary
    model.summary()
    
    # Create a more detailed visualization of the model
    plt.figure(figsize=(15, 10))
    
    # Get layer information
    layers = model.layers
    layer_names = [layer.name for layer in layers]
    layer_types = [layer.__class__.__name__ for layer in layers]
    layer_output_shapes = [layer.output_shape for layer in layers]
    
    # Create a table of layer information
    cell_text = []
    for i in range(len(layers)):
        cell_text.append([
            layer_names[i],
            layer_types[i],
            str(layer_output_shapes[i])
        ])
    
    # Create the table
    plt.table(
        cellText=cell_text,
        colLabels=['Layer Name', 'Layer Type', 'Output Shape'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3, 0.4]
    )
    
    plt.axis('off')
    plt.title('LSTM Traffic Prediction Model Architecture', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/model_details.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model visualizations saved to 'visualizations' directory:")
    print("1. model_architecture.png - Detailed model graph")
    print("2. model_details.png - Layer information table")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'best_model.h5'
    
    visualize_model(model_path) 