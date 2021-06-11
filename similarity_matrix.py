# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:39:50 2021

@author: snigd
"""

import numpy as np
from os import listdir, path
from sklearn import metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# replace with input and output folder
# input folder should only have .npy files
folder = ''
output_folder = ''

# preparing vectors from .npy files
vectors = []
for file in listdir(folder):
    matrix = np.load(path.join(folder, file))
    vector = []
    for i in range(len(matrix[0])):
        vector.append(np.mean(matrix[:,i]))
    
    vector = np.asarray(vector)
    vectors.append(vector)

sample_matrix = np.vstack(vectors)


# correlation distances
distances = metrics.pairwise_distances(sample_matrix, metric="correlation")
ax = plt.subplot()
im = ax.imshow(distances)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.title("Correlation Distances")
plt.savefig(path.join(output_folder, "correlation.png"), dpi=300) 

# cosine distances
distances = metrics.pairwise_distances(sample_matrix, metric="cosine")
ax = plt.subplot()
im = ax.imshow(distances)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.title("Cosine Distances")
plt.savefig(path.join("output_folder", "cosine.png"), dpi=300) # replace for others

# euclidean distances
distances = metrics.pairwise_distances(sample_matrix, metric="euclidean")
ax = plt.subplot()
im = ax.imshow(distances)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.title("Euclidean Distances")
plt.savefig(path.join("output_folder", "euclidean.png"), dpi=300) 