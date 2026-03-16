from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def dimension_PCA(data, input_dimension):
    pca = PCA(n_components=input_dimension)
    data_pca = pca.fit_transform(data)

    #whole_pca = np.zeros((data_UP.shape[0], data_UP.shape[1], input_dimension))
    #print (whole_pca.shape)

    #for i in range(input_dimension):
         #whole_pca[:, :, i] = pca.components_[i].reshape(data_UP.shape[0], data_UP.shape[1])

    #print (whole_pca.shape)

    return data_pca

