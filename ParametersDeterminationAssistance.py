import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# Parameters
PolynomDegree = 2


def determine_max_error_for_polynom_fitting(X, x_vals, y_vals):
    # This function creates plots and information in order to detrmine the maximal threshold for data to fit a polynoial.

        # Cluster
        clustering = DBSCAN(eps=100, min_samples=4).fit(np.concatenate(X))
        # clustering = DBSCAN(eps=50, min_samples=3).fit(X)
        labels = clustering.labels_  # getting the labels
        n_lanes = labels.max()

        # Fit 2nd degree polynom to the data
        for lane in range(n_lanes + 1):
            lane_x_values = x_vals[labels == lane]
            lane_y_values = y_vals[labels == lane]
            polynom = np.polyfit(lane_x_values, lane_y_values, PolynomDegree)
            # Check for inliers (also not within the cluster):
            model = np.poly1d(polynom)
            cluster_errors = np.abs(model(lane_x_values) - lane_y_values)
            no_cluster_errors = np.abs(model(x_vals[labels != lane]) - y_vals[labels != lane])
            print("Cluster ", lane, ": Inliers max error: ", cluster_errors.max(), ", Not inliers min error: ", no_cluster_errors.min())
            plt.plot(cluster_errors, 'o')
            plt.plot(no_cluster_errors, 'o')
            plt.show()


if __name__ == '__main__':
    RawData = pd.read_csv('Data/raw_point.csv')
    X = RawData[['x', 'y']]
    x_vals = X["x"]
    y_vals = X['y']
    determine_max_error_for_polynom_fitting(X, x_vals, y_vals)