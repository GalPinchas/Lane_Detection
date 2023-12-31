import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import Parameters


def plot_raw_data(xy_data):
    x_vals = xy_data['x']
    y_vals = xy_data['y']
    plt.plot(x_vals, y_vals, 'o')
    plt.ylim(max(y_vals), min(y_vals))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Raw data- Lane detection')
    plt.show()


def determine_max_error_for_polynom_fitting(xy_data):
    # This function creates plots and information in order to determine the maximal threshold for data to fit a polynoial
    clustering = DBSCAN(eps=Parameters.cluster_epsilon, min_samples=Parameters.cluster_min_samples).fit(xy_data)
    labels = clustering.labels_  # getting the labels
    n_lanes = labels.max()
    x_vals = xy_data['x']
    y_vals = xy_data['y']
    # Fit 2nd degree polynom to the data
    for lane in range(n_lanes + 1):
        lane_x_values = x_vals[labels == lane]
        lane_y_values = y_vals[labels == lane]
        polynom = np.polyfit(lane_x_values, lane_y_values, Parameters.polynom_degree)
        model = np.poly1d(polynom)
        cluster_errors = np.abs(model(lane_x_values) - lane_y_values)
        no_cluster_errors = np.abs(model(x_vals[labels != lane]) - y_vals[labels != lane])
        print("Cluster ", lane, ": Inliers max error: ", cluster_errors.max(), ", Not inliers min error: ", no_cluster_errors.min())
        plt.plot(cluster_errors, 'o', label='cluster inliers')
        plt.plot(no_cluster_errors, 'o', label='cluster outliers')
        plt.xlabel('Index')
        plt.ylabel('Error of polynomial fit')
        plt.title('Cluster %i- Error values of polynomail fit' %(lane+1))
        plt.show()


def determine_cluster_epsilon(xy_data):
    # This function plots a knee-curve based on k-nearest neighbors in order to estimate Parameter.cluster_epsilon
    nbrs = NearestNeighbors(n_neighbors=Parameters.cluster_min_samples).fit(xy_data)
    distances, indices = nbrs.kneighbors(xy_data)
    k_dist = [x[-1] for x in distances]
    plt.plot(sorted(np.reshape(distances, -1)))
    plt.xlabel('Index')
    plt.ylabel('Distance between K nearest neighbors')
    plt.title('Distance between data points of K nearest neighbors')
    plt.show()


if __name__ == '__main__':
    RawData = pd.read_csv('Data/raw_point.csv')
    xy_data = RawData[['x', 'y']]
    plot_raw_data(xy_data)
    determine_max_error_for_polynom_fitting(xy_data)
    determine_cluster_epsilon(xy_data)