Dog and Cat Image Classifier using a Multi-Layer Perceptron (MLP)
This project demonstrates the development of a multi-layer perceptron (MLP) neural network to classify images of dogs and cats. It focuses on a foundational approach to image classification using scikit-learn, highlighting key machine learning concepts such as data preprocessing, model architecture evaluation, and performance tuning.

Project Highlights
Model Development: An MLP neural network was designed and implemented from scratch using the MLPClassifier from scikit-learn.

Architecture Evaluation: The project explores the impact of different network architectures, including the number of hidden layers and neurons, as well as various activation functions like ReLU and Tanh, on model performance.

Data Preprocessing: Implemented techniques such as image flattening to prepare pixel data for the MLP, and feature scaling to standardize pixel values, ensuring robust model training.

Performance Tuning: The model's robustness was improved through data augmentation techniques like noise augmentation, which helps the model generalize better to unseen images.

Files in this Repository
mlp_classifier.py: The main Python script containing the entire project workflow.

requirements.txt: A list of all necessary Python libraries.

README.md: This file, providing an overview of the project.

How to Run the Project
Clone the Repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install Required Libraries:

pip install -r requirements.txt

Run the Python Script:

python mlp_classifier.py

Requirements
The project requires the following Python libraries, which are listed in requirements.txt:

numpy

scikit-learn

matplotlib

Pillow (PIL)