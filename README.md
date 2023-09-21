# Lane Detection Algorithm

## Goal
This repository is intended to detect lane lines from 2-dimensional raw data, where each lane is modeled by a polynomial 
up to the second degree.

## Method
The algorithm consists of 2 main parts:
1. Clustering algorithm- The data is clustered using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. 
   The algorithm cluster data samples with high spatial density distribution, and separates them based on low-density.  
   DBSCAN requires two parameters:
   1. epsilon: The maximum distance between two samples for one to be considered as in the cluster of the other.
   2. min_points: The minimum number of samples within distance 'epsilon' from it (including the point itself), to ne declared as cluster core point.
2. Polynomial fit- Each cluster is fitted to a 2-nd degree polynomial by the use of polyfit.
   Polyfit aims to minimize the mean squares errors between the data samples and the estimated values.

## Project Structure
The project consist of several files:
1. main.py: The main algorithm that performs the lane detection task. 
The file consists of two functions (cluster_data_points, lane_polynomial_fitting) and takes as input 'Data/raw_points.csv' and imports 'Parameters.py'. 
It output the polynomial equations in the file 'Results.txt', as well as graphs to visualize the data.
2. Data/raw_points.csv: The input file, a 242*3 table, that consists of x-y data samples.
3. ParametersDeterminationAssistance.py: This is a help-file (not imported by main.py), intended to create graphs and data required to determine the values of the following parameters: max_error_for_polynom_fitting, cluster_epsilon.
4. Parameters.py: Definition of numerical parameters.
5. Results.txt: The output file, generated by main.py. The file contains the polynomial equations, determined by the algorithm, along with additional fitting information.

## Requirements
Before beginning, please run:
```
pip install -r requirements.txt
```
