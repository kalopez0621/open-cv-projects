# Functions to read and show images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh 

d0 = pd.read_csv('mnist_train.csv')

print(d0.head(5))  # print first five rows of d0.

# Save the labels into a variable l.
l = d0['label']

# Drop the label feature and store the pixel data in d.
d = d0.drop("label", axis=1)
print(d.shape)
print(l.shape)

def display():
    # Display or plot a number.
    plt.figure(figsize=(7, 7))
    idx = 1

    grid_data = d.iloc[idx].to_numpy().reshape(28, 28)  # reshape from 1D to 2D pixel array
    plt.imshow(grid_data, interpolation="none", cmap="gray")
    plt.show()

    print(l[idx])
# display()

def label():
    # Pick first 15K data-points to work on for time-efficiency.
    labels = l.head(15000)
    data = d.head(15000)

    print("the shape of sample data = ", data.shape)

    # Data-preprocessing: Standardizing the data
    standardized_data = StandardScaler().fit_transform(data)
    print(standardized_data.shape)

    # Find the covariance matrix which is: A^T * A
    covar_matrix = np.matmul(standardized_data.T, standardized_data)

    print("The shape of variance matrix = ", covar_matrix.shape)

    # Calculate all eigenvalues and eigenvectors
    values, vectors = eigh(covar_matrix)

    # Get the indices of the top two eigenvalues
    indices = np.argsort(values)[-2:]  # Get the indices of the top two eigenvalues
    top_vectors = vectors[:, indices]  # Get the corresponding eigenvectors

    print("Shape of eigen vectors = ", top_vectors.shape)
    
    # Converting the eigen vectors into (2, d) shape for ease of further computations
    top_vectors = top_vectors.T

    print("Updated shape of eigen vectors = ", top_vectors.shape)
    # Now top_vectors[0] and top_vectors[1] represent the top two principal eigenvectors.

label()
