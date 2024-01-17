from utils import *

##### FilePath and Globals #####

Images_path = "./Images"
macbeth_7_path = Images_path + "/macbeth_7.mat"
flowers_7_path = Images_path + "/flowers_7.mat"
D65_path = Images_path + "/Illu_D65.mat"
A_Path = Images_path + "/Illuminant_A.mat"
CMF_path = Images_path + "/CMF.mat"
macbeth_31_24_path = Images_path + "/Macbeth_31_24.mat"
image_RGB_path = Images_path + "/Image_RGB.mat"

##### Loading the necessary files and variable #####

macbeth_7 = load_mat(macbeth_7_path)
flowers_7 = load_mat(flowers_7_path)
D65 = load_mat(D65_path)
A = load_mat(A_Path)
CMF = load_mat(CMF_path)
R = load_mat(macbeth_31_24_path)
Image_RGB = load_mat(image_RGB_path) # Reference for MACBETH in RGB space

num_levels = 31
base_depth = 7

centers = np.array([
    [60, 50], [60, 125], [60, 200], [60, 275], [60, 350], [60, 425],
    [134, 50], [134, 125], [134, 200], [134, 275], [134, 350], [134, 425],
    [208, 50], [208, 125], [208, 200], [208, 275], [208, 350], [208, 425],
    [282, 50], [282, 125], [282, 200], [282, 275], [282, 350], [282, 425]
])

RGB_XYZ = np.array(
            [
                [0.429, 0.343, 0.178],
                [0.222, 0.7070, 0.071],
                [0.0190, 0.132, 0.939]
            ]
        )

XYZ_RGB = np.linalg.inv(RGB_XYZ)

##### MANIPULATION 1 #####

### LINEAR INTERPOLATION ###

# Save path
save_path_m1 = "saved/manipulation_1"

nodes = np.arange(base_depth) / (base_depth - 1)

# Normalize and Inverse the D65 matrix illuminant
D65 = D65 / np.max(D65)
D65 = 1.0 / D65

# Échantillonnage et interpolations
t_values = np.arange(num_levels) / (num_levels - 1)

linear_macbeth_31 = np.transpose(np.array([linear_interpolation(macbeth_7, nodes, t) for t in t_values]))
linear_flowers_31 = np.transpose(np.array([linear_interpolation(flowers_7, nodes, t) for t in t_values]))

linear_macbeth_31_corrected = linear_macbeth_31 * D65.T
linear_flowers_31_corrected = linear_flowers_31 * D65.T

# Save the results
save_bands(save_path_m1 + "/macbeth/linear", "linear_macbeth_31", 400, 10, linear_macbeth_31)
save_bands(save_path_m1 + "/macbeth/linear", "corrected_linear_macbeth_31", 400, 10, linear_macbeth_31_corrected)

save_bands(save_path_m1 + "/flowers/linear", "linear_flowers_31", 400, 10, linear_flowers_31)
save_bands(save_path_m1 + "/flowers/linear", "corrected_linear_flowers_31", 400, 10, linear_flowers_31_corrected)


### HERMITE INTERPOLATION ###

hermite_macbeth_31 = np.transpose(np.array([hermite_interpolation(macbeth_7, nodes, t) for t in t_values]))
hermite_flowers_31 = np.transpose(np.array([hermite_interpolation(flowers_7, nodes, t) for t in t_values]))

hermite_macbeth_31_corrected = hermite_macbeth_31 * D65.T
hermite_flowers_31_corrected = hermite_flowers_31 * D65.T

# Save the results
save_bands(save_path_m1 + "/macbeth/Hermite", "hermite_macbeth_31", 400, 10, hermite_macbeth_31)
save_bands(save_path_m1 + "/macbeth/Hermite", "corrected_hermite_macbeth_31", 400, 10, hermite_macbeth_31_corrected)

save_bands(save_path_m1 + "/flowers/Hermite", "hermite_flowers_31", 400, 10, hermite_flowers_31)
save_bands(save_path_m1 + "/flowers/Hermite", "corrected_hermite_flowers_31", 400, 10, hermite_flowers_31_corrected)


### SPLINE INTERPOLATION ###

spline_macbeth_31 = np.transpose(np.array([spline_interpolation(macbeth_7, nodes, t) for t in t_values]))
spline_flowers_31 = np.transpose(np.array([spline_interpolation(flowers_7, nodes, t) for t in t_values]))

spline_macbeth_31_corrected = spline_macbeth_31 * D65.T
spline_flowers_31_corrected = spline_flowers_31 * D65.T

# Save the results
save_bands(save_path_m1 + "/macbeth/spline", "spline_macbeth_31", 400, 10, spline_macbeth_31)
save_bands(save_path_m1 + "/macbeth/spline", "corrected_spline_macbeth_31", 400, 10, spline_macbeth_31_corrected)

save_bands(save_path_m1 + "/flowers/spline", "spline_flowers_31", 400, 10, spline_flowers_31)
save_bands(save_path_m1 + "/flowers/spline", "corrected_spline_flowers_31", 400, 10, spline_flowers_31_corrected)

##### MANIPULATION 2 #####

# Save path
save_path_m2 = "saved/manipulation_2"

# Create the vector D
D = np.zeros((7,24))

# Compute transition matrix Q
Q = inversion_indirecte_compute_Q(macbeth_7, R, centers)

inversion_flowers_31 = np.dot(flowers_7.astype(Q.dtype), Q)
inversion_macbeth_31 = np.dot(macbeth_7.astype(Q.dtype), Q)

save_bands(save_path_m2 + "/flowers", "inversion_flowers_31", 400, 10, inversion_flowers_31)
save_bands(save_path_m2 + "/macbeth", "inversion_macbeth_31", 400, 10, inversion_macbeth_31)

##### MANIPULATION 3 #####

# Save path
save_path_m3 = "saved/manipulation_3"

#---- PART 1 ----#

# Do the 3 bands BGRs images for macbeth and flowers
BGR_flowers = np.zeros((flowers_7.shape[0], flowers_7.shape[1], 3))
BGR_macbeth = np.zeros((macbeth_7.shape[0], macbeth_7.shape[1], 3))

BGR_flowers[:,:,0] = inversion_flowers_31[:,:, 5]   #450nm 
BGR_flowers[:,:,1] = inversion_flowers_31[:,:,15]   #550nm
BGR_flowers[:,:,2] = inversion_flowers_31[:,:,22]   #620nm

BGR_macbeth[:,:,0] = inversion_macbeth_31[:,:, 5]   #450nm
BGR_macbeth[:,:,1] = inversion_macbeth_31[:,:,15]   #550nm
BGR_macbeth[:,:,2] = inversion_macbeth_31[:,:,22]   #620nm

BGR_macbeth_norm = normalize_to_image(BGR_macbeth)
BGR_flowers_norm = normalize_to_image(BGR_flowers)

save_bands(save_path_m3 + "/flowers", "BGR_flowers", 1, 1, BGR_flowers_norm)
save_bands(save_path_m3 + "/macbeth", "BGR_macbeth", 1, 1, BGR_macbeth_norm)

cv2.imwrite(save_path_m3 + "/flowers/RGB_flowers.png", BGR_flowers_norm)
cv2.imwrite(save_path_m3 + "/macbeth/RGB_macbeth.png", BGR_macbeth_norm)

#---- PART 2 ----#

# Apply CMF matrix
inversion_flowers_31_CMF = np.dot( inversion_flowers_31, CMF)
inversion_macbeth_31_CMF = np.dot( inversion_macbeth_31, CMF)

# Apply XYZ_RGB matrix
inversion_flowers_31_CMF_XYZ = np.abs( np.dot(inversion_flowers_31_CMF, XYZ_RGB.T) * 0.00169 )
inversion_macbeth_31_CMF_XYZ = np.abs( np.dot(inversion_macbeth_31_CMF, XYZ_RGB.T) * 0.00169 )

# Normalization to image format
inversion_flowers_31_CMF_norm = cv2.cvtColor( normalize_to_image(inversion_flowers_31_CMF), cv2.COLOR_RGB2BGR )
inversion_flowers_31_CMF_XYZ_norm = cv2.cvtColor( normalize_to_image(inversion_flowers_31_CMF_XYZ), cv2.COLOR_RGB2BGR )

inversion_macbeth_31_CMF_norm = cv2.cvtColor( normalize_to_image(inversion_macbeth_31_CMF), cv2.COLOR_RGB2BGR )
inversion_macbeth_31_CMF_XYZ_norm = cv2.cvtColor( normalize_to_image(inversion_macbeth_31_CMF_XYZ), cv2.COLOR_RGB2BGR )

cv2.imwrite(save_path_m3 + "/flowers/CMF_flowers.png", inversion_flowers_31_CMF_norm )
cv2.imwrite(save_path_m3 + "/flowers/CMF_XYZ_flowers.png", inversion_flowers_31_CMF_XYZ_norm )

cv2.imwrite(save_path_m3 + "/macbeth/CMF_macbeth.png", inversion_macbeth_31_CMF_norm )
cv2.imwrite(save_path_m3 + "/macbeth/CMF_XYZ_macbeth.png", inversion_macbeth_31_CMF_XYZ_norm )

##### Stastiques #####

# Save path
save_path_stats = "saved/statistics"

# convert the reference to BGR
Image_BGR = cv2.cvtColor( Image_RGB, cv2.COLOR_RGB2BGR )

# Saving reference image for macbeth calibration
cv2.imwrite(save_path_stats + "/macbeth.png", Image_BGR )

# MSE scores in BGR space compared to the reference
mse_macbeth_BGR = MSE(Image_BGR ,BGR_macbeth_norm)
mse_macbeth_CMF = MSE(Image_BGR ,inversion_macbeth_31_CMF_norm)
mse_macbeth_XYZ = MSE(Image_BGR ,inversion_macbeth_31_CMF_XYZ_norm)

# Delta scores in Lab space
DLab_macbeth_BGR = Delta_Lab(Image_BGR ,BGR_macbeth_norm)
DLab_macbeth_CMF = Delta_Lab(Image_BGR ,inversion_macbeth_31_CMF_norm)
DLab_macbeth_XYZ = Delta_Lab(Image_BGR ,inversion_macbeth_31_CMF_XYZ_norm)

# HSV score (compare the mean HSV differences)
reference_HSV = cv2.cvtColor(Image_BGR, cv2.COLOR_BGR2HSV)
macbeth_HSV = cv2.cvtColor(BGR_macbeth_norm, cv2.COLOR_BGR2HSV)
macbeth_HSV_CMF = cv2.cvtColor(inversion_macbeth_31_CMF_norm, cv2.COLOR_BGR2HSV)
macbeth_HSV_XYZ = cv2.cvtColor(inversion_macbeth_31_CMF_XYZ_norm, cv2.COLOR_BGR2HSV)

HSV_Score_3B  = MSE(reference_HSV, macbeth_HSV)
HSV_Score_CMF = MSE(reference_HSV, macbeth_HSV_CMF)
HSV_Score_XYZ = MSE(reference_HSV, macbeth_HSV_XYZ)

#Split H,S,V MSE scores (detailled scores)

# Hue scores
H_Score_3B  = MSE(reference_HSV[0], macbeth_HSV[0])
H_Score_CMF = MSE(reference_HSV[0], macbeth_HSV_CMF[0])
H_Score_XYZ = MSE(reference_HSV[0], macbeth_HSV_XYZ[0])

# Saturation scores
S_Score_3B  = MSE(reference_HSV[1], macbeth_HSV[1])
S_Score_CMF = MSE(reference_HSV[1], macbeth_HSV_CMF[1])
S_Score_XYZ = MSE(reference_HSV[1], macbeth_HSV_XYZ[1])

# Value scores
V_Score_3B  = MSE(reference_HSV[2], macbeth_HSV[2])
V_Score_CMF = MSE(reference_HSV[2], macbeth_HSV_CMF[2])
V_Score_XYZ = MSE(reference_HSV[2], macbeth_HSV_XYZ[2])

# Plot the data and save it

# Create a single figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7))
colors = ['red', 'green', 'blue']

# Plotting MSE scores
ax1.bar(['BGR', 'CMF', 'XYZ'], [mse_macbeth_BGR, mse_macbeth_CMF, mse_macbeth_XYZ], color = colors)
ax1.set_title('MSE Scores Comparison (lower is better)')
ax1.set_ylabel('MSE Score')

# Plotting Delta Lab scores
ax2.bar(['BGR', 'CMF', 'XYZ'], [DLab_macbeth_BGR, DLab_macbeth_CMF, DLab_macbeth_XYZ], color = colors)
ax2.set_title('Delta Lab Scores Comparison (lower is better)')
ax2.set_ylabel('Delta Lab Score')

# Plotting HSV_MSE scores
ax3.bar(['BGR', 'CMF', 'XYZ'], [HSV_Score_3B, HSV_Score_CMF, HSV_Score_XYZ], color = colors)
ax3.set_title('HSV MSE Scores Comparison (lower is better)')
ax3.set_ylabel('HSV MSE Score')

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure to a file
plt.savefig(f"{save_path_stats}/comparison_plot.png")

# Show the combined plot
plt.show()


# Create a single figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7))

# Plotting MSE scores
ax1.bar(['BGR', 'CMF', 'XYZ'], [H_Score_3B, H_Score_CMF, H_Score_XYZ], color = colors)
ax1.set_title('Hue MSE Scores Comparison (lower is better)')
ax1.set_ylabel('Hue MSE Score')

# Plotting Delta Lab scores
ax2.bar(['BGR', 'CMF', 'XYZ'], [S_Score_3B, S_Score_CMF, S_Score_XYZ], color = colors)
ax2.set_title('Saturation MSE Scores Comparison (lower is better)')
ax2.set_ylabel('Saturation MSE Score')

# Plotting HSV_MSE scores
ax3.bar(['BGR', 'CMF', 'XYZ'], [V_Score_3B, V_Score_CMF, V_Score_XYZ], color = colors)
ax3.set_title('Value MSE Scores Comparison (lower is better)')
ax3.set_ylabel('Value MSE Score')

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure to a file
plt.savefig(f"{save_path_stats}/comparison_plot_HSV.png")

# Show the combined plot
plt.show()

