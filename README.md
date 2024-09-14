# Linear Regression Using PyTorch

This project demonstrates how to implement a linear regression model using PyTorch. The model is designed to predict house sale prices based on various features from a dataset.

## Overview

The project includes:

- Loading and preparing the dataset.
- Defining a linear regression model with PyTorch.
- Training the model and evaluating its performance.
- Visualizing training and testing losses over epochs.

## Requirements

Ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `torch`
- `scikit-learn`
- `matplotlib`

You can install the required libraries using pip:

```bash
pip install pandas numpy torch scikit-learn matplotlib
```

## Data Preparation

### 1. Load and Inspect Data

1. **Load the Dataset**
   - Load the dataset into a Pandas DataFrame.

2. **Inspect the Data**
   - Use methods such as `describe()`, `shape`, `isnull().sum()`, and `dtypes` to understand the dataset's structure and check for missing values.

3. **Preprocess the Data**
   - Encode categorical variables using `LabelEncoder`.
   - Split the dataset into features and target variable.

### 2. Data Splitting

1. **Train-Test Split**
   - Split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.

2. **Convert to PyTorch Tensors**
   - Convert the features and target variable to PyTorch tensors.

## Model Definition

### 1. Define the Model Architecture

1. **Linear Regression Model**
   - **Layers:** 
     - `Linear1`: Fully connected layer with input dimension to hidden layer (10 neurons).
     - `Linear2`: Another fully connected layer with hidden layer to hidden layer (10 neurons).
     - `Linear3`: Final fully connected layer from hidden layer to output dimension (1 neuron).
   - **Activations:**
     - `ReLU`: Applied after each linear layer.
     - `Sigmoid`: Applied to the final output (optional based on your needs).

```python
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.linear(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        return out
```

## Training

### 1. Initialize Model, Loss, and Optimizer

1. **Initialize Model**
   - Create an instance of the `LinearRegression` model with input dimension (number of features) and output dimension (1 for sale price).

2. **Define Loss Function**
   - Use `L1Loss` for regression tasks.

3. **Set Up Optimizer**
   - Use `Adam` optimizer for training with a learning rate of 0.001.

### 2. Training Loop

1. **Training Process**
   - Iterate over epochs, compute the loss, and update model parameters.
   - Track training and testing losses for each epoch.

```python
import matplotlib.pyplot as plt

plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss over Epochs')
plt.show()
```

## Example Usage

To train the model and visualize the losses:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define and initialize model
model = LinearRegression(input_dim=4, output_dim=1)
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
test_losses = []
epochs = []

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test)
        test_loss = loss_fn(y_pred, y_test)
        test_losses.append(test_loss.item())
    
    epochs.append(epoch)

# Plot training and testing losses
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()
```

## Files

- `linear_regression_pytorch.ipynb`: Jupyter Notebook containing the code for model training and evaluation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- PyTorch for the deep learning framework.
- Scikit-learn for data preprocessing utilities.
- Matplotlib for plotting.
```

This `README.md` file provides a structured and detailed overview of your PyTorch linear regression project, making it easy for others to understand and replicate your work.
