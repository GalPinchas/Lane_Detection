import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import os

import Parameters

# Delete the output file "Results.txt" (if exists):
if os.path.exists("Results.txt"):
    os.remove("Results.txt")


def cluster_data_points(xy_data):
    # This function partition the input data xy_data into clusters, by the DBSCAN algorithm,
    # and output the labels vector of the input data.
    clustering = DBSCAN(eps=Parameters.cluster_epsilon, min_samples=Parameters.cluster_min_samples).fit(xy_data)
    labels = clustering.labels_ # labels: 0...n-1 (for n clusters), -1=outlier
    return(labels)


def lane_polynomial_fitting(x, y, inlier_indices):
    # This function fits a polynomial (with degree specified by Parameters.polynom_degree)
    # to the vectors [x,y], only in the indices inlier_indices.
    lane_x_values = x[inlier_indices]
    lane_y_values = y[inlier_indices]
    polynom = np.polyfit(lane_x_values, lane_y_values, Parameters.polynom_degree)
    model = np.poly1d(polynom)
    return model, lane_x_values


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    RawData = pd.read_csv('Data/raw_point.csv')
    xy_data = RawData[['x', 'y']]
    x_vals = xy_data['x']
    y_vals = xy_data['y']

    # Cluster the data:
    labels = cluster_data_points(xy_data)
    n_lanes = labels.max()

    # Fit 2nd degree polynom to each of the clusters:
    for lane in range(n_lanes+1):
        if sum(labels == lane) > Parameters.min_data_points_for_polynom_fitting:
            model, lane_x_values = lane_polynomial_fitting(x_vals, y_vals, labels == lane)
            # Check for more inliers (also not within the cluster):
            residuals = model(x_vals) - y_vals
            polynom_inliers = np.abs(residuals) < Parameters.max_error_for_polynom_fitting
            num_of_polynom_inliers = sum(polynom_inliers)
            if num_of_polynom_inliers > Parameters.min_data_points_for_polynom_fitting:
                # Check if there are more/less data points in the DB to match the polynomial:
                if num_of_polynom_inliers != len(lane_x_values):
                    model, lane_x_values = lane_polynomial_fitting(x_vals, y_vals, polynom_inliers)
                    # Update labels according to the new inliers:
                    labels[polynom_inliers] = lane
                    org_polynom_inliers = labels == lane
                    labels[org_polynom_inliers & np.invert(polynom_inliers)] = -1

                # Plotting the lane:
                LaneLine_x = np.linspace(lane_x_values.min(), lane_x_values.max(), 1000)
                plt.plot(LaneLine_x, model(LaneLine_x), color="black")
                # Print to file:
                print('Lane border %i:\nPolynomial equation: %f*x^2 + %f*x + %f' %(lane+1, model.coef[0], model.coef[1], model.coef[2]), file=open('Results.txt', 'a'))
                print('Number of points for polynomial fitting: %i, With average residuals (abs) of: %f \n' %(num_of_polynom_inliers, np.mean(np.abs(residuals[polynom_inliers]))), file=open('Results.txt', 'a'))


    # Plot the clusters
    plt.scatter(x_vals, y_vals, c=labels, cmap="plasma")  # plotting the clusters
    plt.xlabel("X")  # X-axis label
    plt.ylabel("Y")  # Y-axis label
    plt.title("Lane Detection - Clusters & Polynomial Fits")
    plt.ylim(max(y_vals), min(y_vals))
    plt.show()