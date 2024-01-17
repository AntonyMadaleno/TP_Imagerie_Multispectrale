import matplotlib.pyplot as plt
import scipy.io as sc
import numpy as np
import cv2
from scipy.interpolate import CubicSpline

#interpolation function (Spline)
def spline_interpolation(data, nodes, t):
    """
    Spline Interpolation using Cubic Splines

    Parameters:
    - data: NumPy array of float values.
    - nodes: NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t: Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """
    spline = CubicSpline(nodes, data, axis=2)
    return normalize_to_image(spline(t))

#interpolation function (Hermite)
def hermite_interpolation(data, nodes, t):
    """
    Hermite Interpolation

    Parameters:
    - data:     NumPy array of float values.
    - nodes:    NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t:        Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """

    n = len(nodes)
    result = 0.0

    for i in range(n):
        # Compute the Hermite basis functions
        basis = 1.0
        for j in range(n):
            if i != j:
                basis *= (t - nodes[j]) / (nodes[i] - nodes[j])

        # Update the result with the contribution of the current data point
        result += data[:,:,i] * basis

    return normalize_to_image(result)

#interpolation function (Linear)
def linear_interpolation(data, nodes, t):
    """
    Linear Interpolation

    Parameters:
    - data:     NumPy array of float values.
    - nodes:    NumPy array of float values indicating the position of each point in the interval [0.0, 1.0].
    - t:        Value between 0 and 1 representing the position where interpolation is needed.

    Returns:
    - Interpolated value
    """

    n = len(nodes)
    result = 0.0

    for i in range(n-1):
        if (nodes[i] <= t):
            result = data[:,:,i] * abs( ( nodes[i+1] - t ) / ( nodes[i+1] - nodes[i] ) ) + data[:,:,i+1] * abs( ( nodes[i] - t ) / ( nodes[i+1] - nodes[i] ) )

    return normalize_to_image(result)

# Loader for the .mat files
def load_mat(file_path : str):

    mat = sc.loadmat(file_path)

    data = np.zeros(1)

    for key in mat:
        if(type(mat[key]) == type(data)):
            data = mat[key]
    
    return data

# Save each bands of the image as grayscale in given folder
def save_bands(file_path : str, filename : str, w_min : int, step : int, data):

    if (data.dtype != np.uint8):
        data = normalize_to_image(data)

    for i in range(data.shape[2]):
        cv2.imwrite(f"{file_path}/{filename}_{w_min + i*step}nm.png", data[:,:,i].astype(np.uint8))

# Compute the transition matrix Q for indirect inversion
def inversion_indirecte_compute_Q(calibration, R, centers):

    #Create the vector D
    D = np.zeros((7,24))

    k = 0
    for c in centers:
        x = c[0]
        y = c[1]

        #select the patch to evaluate
        patch = calibration[ x - 20:x + 20, y - 20:y + 20, :]
        
        #Calculate Mean values for each patch at each 
        for i in range( patch.shape[2] ):
            D[i,k] = np.mean(patch[:,:,i])

        k = k+1

    # Calculate Q
    Q = np.linalg.lstsq(D.T, R, rcond= None )[0]

    return Q

# Normalize the data to image format (uint8 [0,255])
def normalize_to_image(data):
    return ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

# Calculate MSE score in between an image and a given reference of same size
def MSE(reference, image):

    if (reference.shape != image.shape):
        print(f"Error on calculating MSE, the given images aren't of same size.\nReference is size : {reference.shape}\nImage is size : {image.shape}")
        return -1
    
    return np.mean( np.square(image.astype(np.double) - reference.astype(np.double)) )

# Compute the DeltaE in Lab space given an image and its reference
def Delta_Lab(reference, image):

    if (reference.shape != image.shape):
        print(f"Error on calculating Delta_Lab, the given images aren't of same size.\nReference is size : {reference.shape}\nImage is size : {image.shape}")
        return -1
    
    
    # Convert BGR to Lab color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2Lab)

    return cv2.norm(image_lab, reference_lab, cv2.NORM_L2)