# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

import Parameters

def lane_polynomial_fitting(x, y, inlier_indices):
    lane_x_values = x[inlier_indices]
    lane_y_values = y[inlier_indices]
    polynom = np.polyfit(lane_x_values, lane_y_values, Parameters.polynom_degree)
    model = np.poly1d(polynom)
    return model, lane_x_values


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    RawData = pd.read_csv('Data/raw_point.csv')
    #print(RawData)
    xy_data = RawData[['x', 'y']]
    x_vals = xy_data['x']
    y_vals = xy_data['y']
    #plt.plot(x_vals,y_vals,'o')
    #plt.show()

    # Cluster
    clustering = DBSCAN(eps=Parameters.cluster_epsilon, min_samples=Parameters.cluster_min_samples).fit(xy_data)
    #clustering = DBSCAN(eps=50, min_samples=3).fit(X)
    labels = clustering.labels_  # getting the labels
    n_lanes = labels.max()
    print(n_lanes+1)
    print("0=", sum(labels == 0), "1=", sum(labels == 1), "2=", sum(labels == 2))

    # Fit 2nd degree polynom to the data
    for lane in range(n_lanes+1):
        if sum(labels == lane) > Parameters.min_data_points_for_polynom_fitting:
            model, lane_x_values = lane_polynomial_fitting(x_vals, y_vals, labels == lane)
            # Check for more inliers (also not within the cluster):
            polynom_inliers = np.abs(model(x_vals) - y_vals) < Parameters.max_error_for_polynom_fitting
            if sum(polynom_inliers) != len(lane_x_values):
                model, lane_x_values = lane_polynomial_fitting(x_vals, y_vals, polynom_inliers)
                # Update labels according to the new inliers:
                labels[polynom_inliers] = lane
                org_polynom_inliers = labels == lane
                labels[org_polynom_inliers & np.invert(polynom_inliers)] = -1

            # Plotting the lane:
            LaneLine_x = np.linspace(lane_x_values.min(), lane_x_values.max(), 1000)
            plt.plot(LaneLine_x, model(LaneLine_x), color= "black")

    # Plot the clusters
    plt.scatter(x_vals, y_vals, c=labels, cmap="plasma")  # plotting the clusters
    plt.xlabel("X")  # X-axis label
    plt.ylabel("Y")  # Y-axis label
    plt.title("DBSCAN - Epsilon=50, Min num of samples = 3")
    plt.ylim(max(y_vals), min(y_vals))
    plt.show()

# Add:
# print to figure: For each lane: no. of samples, the polynom, the goodness of fit
# print to file: Results
# function that describes hpw I chose epsilon, n_points
# function that plots the raw data
# after each Polynom finding: check if other points in the data correspond to it. Maybe 2 clusters belong to the same lane.