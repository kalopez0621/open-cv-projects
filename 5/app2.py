# Functions to read and show images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.linalg import eigh
import seaborn as sns
from sklearn.manifold import TSNE  # Importing TSNE directly

# Load the dataset
d0 = pd.read_csv('mnist_train.csv')

print(d0.head(5))  # Print first five rows of d0.

# Save the labels into a variable l.
l = d0['label']

# Drop the label feature and store the pixel data in d.
d = d0.drop("label", axis=1)

print(d.shape)
print(l.shape)

# Display or plot a number.
plt.figure(figsize=(7, 7))
idx = 1

grid_data = d.iloc[idx].to_numpy().reshape(28, 28)  # Reshape from 1D to 2D pixel array
plt.imshow(grid_data, interpolation="none", cmap="gray")
plt.show()

print(l[idx])

# Pick first 15K data-points to work on for time-efficiency.
labels = l.head(15000)
data = d.head(15000)

print("The shape of sample data = ", data.shape)

# Data-preprocessing: Standardizing the data
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)

# Find the covariance matrix which is: A^T * A
sample_data = standardized_data

# Matrix multiplication using numpy
covar_matrix = np.matmul(sample_data.T, sample_data)

print("The shape of variance matrix = ", covar_matrix.shape)

# Finding the top two eigenvalues and corresponding eigenvectors for projecting onto a 2D space.
# Calculate all eigenvalues and eigenvectors
values, vectors = eigh(covar_matrix)

# Specifying the indices of the top two eigenvalues
top_indices = [782, 783]
top_values = values[top_indices]
top_vectors = vectors[:, top_indices]

print("Top eigenvalues (indices 782 and 783): ", top_values)
print("Shape of eigen vectors = ", top_vectors.shape)

# Converting the eigen vectors into (2, d) shape for ease of further computations.
top_vectors = top_vectors.T  # Transposing to (2, d) format

print("Updated shape of eigen vectors = ", top_vectors.shape)

# Projecting the original data sample on the plane formed by two principal eigen vectors
new_coordinates = np.matmul(sample_data, top_vectors.T)  # Use top_vectors for projection

print("Resultant new data points' shape ", sample_data.shape, "X", top_vectors.T.shape, " = ", new_coordinates.shape)

# Appending labels to the 2D projected data
new_coordinates = np.hstack((new_coordinates, labels.values.reshape(-1, 1)))  # Ensure labels are a column

# Creating a new DataFrame for plotting the labeled points
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
print(dataframe.head())

# Example data for plotting
df = pd.DataFrame()
df['1st'] = [-5.558661, -5.043558, 6.193635, 19.305278]
df['2nd'] = [-1.558661, -2.043558, 2.193635, 9.305278]
df['label'] = [1, 2, 3, 4]

# Using FacetGrid to plot
sns.FacetGrid(df, hue="label", height=6).map(plt.scatter, '1st', '2nd').add_legend()
plt.show()

sns.scatterplot(x="1st", y="2nd", hue="label", data=df)

# Plotting the 2D data points with seaborn
sns.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()

sns.scatterplot(x="1st_principal", y="2nd_principal", legend="full", hue="label", data=dataframe)

# PCA using ScikitLearn
pca = decomposition.PCA(n_components=2)  # Setting the number of components

# Fit and transform the data
pca_data = pca.fit_transform(sample_data)

# PCA reduced will contain the 2D projects of simple data.
print("Shape of pca_reduced data: ", pca_data.shape)

# Attaching the label for each 2D data point 
pca_data = np.vstack((pca_data.T, labels)).T

# Creating a new DataFrame which helps us in plotting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sns.FacetGrid(pca_df, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()

# PCA for dimensionality reduction
pca = decomposition.PCA(n_components=784)  # Set to 784 for full dimensionality
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(figsize=(6, 4))
plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# t-SNE using Scikit-learn
# Picking the top 1000 points as t-SNE takes a lot of time for 15K points
data_1000 = standardized_data[0:1000, :]
labels_1000 = labels[0:1000]

model = TSNE(n_components=2, random_state=0)
# Configuring the parameters
# The number of components = 2
# Default perplexity = 30
# Default learning rate = 200
# Default maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_1000)

# Creating a new DataFrame which helps us in plotting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Plotting the result of t-SNE
sns.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

# Experiment with different parameters
model = TSNE(n_components=2, random_state=0, perplexity=50)
tsne_data = model.fit_transform(data_1000)

# Creating a new DataFrame which helps us in plotting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Plotting the result of t-SNE
sns.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50')
plt.show()

model = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=5000)
tsne_data = model.fit_transform(data_1000)

# Creating a new DataFrame which helps us in plotting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Plotting the result of t-SNE
sns.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50, n_iter=5000')
plt.show()

model = TSNE(n_components=2, random_state=0, perplexity=2)
tsne_data = model.fit_transform(data_1000)

# Creating a new DataFrame which helps us in plotting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Plotting the result of t-SNE
sns.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 2')
plt.show()

# Run the same analysis using 42K points with various values of perplexity and iterations.
# If you use all of the points, you can expect plots like this blog below:
# http://colah.github.io/posts/2014-10-Visualizing-MNIST/

