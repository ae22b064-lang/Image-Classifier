# mlp_classifier.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# --- Data Generation (Simulating a dataset) ---
# In a real project, you would load your images from a directory.
# This code simulates the process by creating dummy image data.

def create_dummy_images(num_images=100, img_size=32):
    """
    Creates a dummy dataset of flattened images with 'dog' and 'cat' labels.
    A 'dog' image will have a higher average pixel value than a 'cat' image,
    creating a simple, discernible pattern for the MLP to learn.
    """
    X = []  # To store the image data (features)
    y = []  # To store the labels
    
    for i in range(num_images):
        if i % 2 == 0:  # Create a 'dog' image
            # 'Dog' images have higher pixel values
            img = np.random.randint(150, 256, size=(img_size, img_size, 3))
            label = 1  # 1 for dog
        else:  # Create a 'cat' image
            # 'Cat' images have lower pixel values
            img = np.random.randint(0, 100, size=(img_size, img_size, 3))
            label = 0  # 0 for cat
        
        # Flatten the image into a 1D vector (Crucial for MLP)
        flattened_img = img.flatten()
        X.append(flattened_img)
        y.append(label)

    return np.array(X), np.array(y)

# Generate a synthetic dataset
print("Generating synthetic image data...")
X, y = create_dummy_images(num_images=200, img_size=32)
print("Data generation complete.")
print(f"Shape of image data (features): {X.shape}")
print(f"Shape of labels: {y.shape}")

# --- Data Preprocessing and Splitting ---

# Employ feature scaling to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled using StandardScaler.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- Model Development and Architecture Evaluation ---

# We'll evaluate two different MLP architectures
architectures = {
    'Simple_MLP': (100,),  # A single hidden layer with 100 neurons
    'Deeper_MLP': (200, 100) # Two hidden layers with 200 and 100 neurons
}

# We'll also evaluate two different activation functions
activation_functions = ['relu', 'tanh']

results = []

for name, hidden_layers in architectures.items():
    for activation in activation_functions:
        print(f"\n--- Training {name} with {activation} activation ---")
        
        # Create the MLPClassifier model with specific architecture and activation
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='adam',
            max_iter=300,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results.append({
            'model_name': name,
            'activation': activation,
            'accuracy': accuracy,
            'report': report
        })

# --- Displaying Results ---
print("\n" + "="*50)
print("Model Performance Summary")
print("="*50)
for result in results:
    print(f"\nModel: {result['model_name']} | Activation: {result['activation']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    print(result['report'])

print("--- Plotting a comparison of model accuracies ---")
model_names = [f"{r['model_name']}\n({r['activation']})" for r in results]
accuracies = [r['accuracy'] for r in results]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['skyblue', 'salmon', 'skyblue', 'salmon'])
plt.ylim(0, 1)
plt.title('MLP Model Performance Comparison')
plt.xlabel('Model Architecture and Activation Function')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--')
plt.show()

print("\nScript finished.")
