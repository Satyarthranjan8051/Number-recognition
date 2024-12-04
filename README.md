
# Handwritten Digit Recognition

This project focuses on recognizing handwritten digits using Convolutional Neural Networks (CNNs) with data preprocessing, augmentation, and model training using TensorFlow/Keras.

## Project Overview
- **Dataset:** MNIST-like dataset of handwritten digits.
- **Model Architecture:** CNN with Conv2D, MaxPooling, Dropout, and Dense layers.
- **Data Augmentation:** Applied to enhance model performance.
- **Training:** Trained over 30 epochs with validation.

## Requirements
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras

## Setup Instructions
1. **Install Required Libraries:**  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

2. **Load the Dataset:**  
   Place the `train.csv`, `test.csv`, and `sample_submission.csv` files in the specified directories.

3. **Run the Code:**  
   Execute the main script to preprocess, train, and evaluate the model.

## Model Summary
- **Conv2D Layer:** 32 filters, kernel size (5x5)
- **MaxPooling:** Pool size (2x2)
- **Dropout:** Rate 0.25
- **Dense Layers:** 256 neurons, followed by a 10-neuron output layer with softmax activation

## Training and Validation Accuracy
The model achieved a validation accuracy of approximately **98.88%**.

## Sample Code Snippet
```python
# Define the model
model = keras.Sequential([
    Conv2D(32, (5,5), padding='Same', activation='relu', input_shape=(28,28,1)),
    MaxPool2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax'),
])
```

## Results Visualization
Example plots of the training data are included, displaying sample digits with corresponding labels.

---

**Author:** Satyarth Ranjan  
**License:** MIT
