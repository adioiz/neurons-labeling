import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import numpy as np

def plot_pixels_on_image(csv_file, image_file, frame_num):
    """
    Plots the pixels from the CSV table on top of the actual neuron frame image.
    """
    # 1. Load Coordinates from CSV
    df = pd.read_csv(csv_file)
    frame_data = df[df['frame'] == frame_num]
            
    x = frame_data['x'].values
    y = frame_data['y'].values
    
    # 2. Load the Image Stack and Select Frame
    img_stack = tifffile.imread(image_file)
    img_frame = img_stack[frame_num]
    
    # 3. Create Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(img_frame, cmap='gray')
    plt.scatter(x, y, s=3, c='red', marker='s', label='Table Coordinates')
    
    plt.title(f"Predicted on Frame {frame_num}")
    plt.axis('off')
    plt.legend(loc='upper right')
    
    plt.show()


plot_pixels_on_image('A31_C2_labels.csv', 'A31_C2.tif', 60)