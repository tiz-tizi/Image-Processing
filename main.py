import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import medfilt2d
from scipy.ndimage import label
from skimage.measure import regionprops
from scipy.spatial import KDTree
from RANSAC import RANSAC


######### 1.Getting and reading the data #########
i = 2  # Choose the example number
data = sio.loadmat(f'example{i}kinect.mat')

# Access amplitude image
if i == 4:
    amplitude_image = data[f'amplitudes{i - 1}']
else:
    amplitude_image = data[f'amplitudes{i}']

# Visualize amplitude image
plt.imshow(amplitude_image)
plt.title("Amplitude Image")
# plt.savefig(f'amplitude_image_{i}.png')
plt.show()

# Access point cloud
point_cloud = data[f'cloud{i}']

# Visualize point cloud
subsampled_point_cloud = point_cloud[::30]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(subsampled_point_cloud[:, :, 0], subsampled_point_cloud[:, :, 1], subsampled_point_cloud[:, :, 2])
ax.set_title("Point Cloud")
# ax.view_init(10,0)
# plt.savefig(f'point_cloud_{i}.png')
plt.show()

# Access distance image
distance_image = data[f'distances{i}']

# plot the hitogram of distance_image
plt.hist(distance_image.flatten(), bins=100)
plt.title("Distance Image Histogram")
plt.xlabel("Distance")
plt.ylabel("Frequency")
# plt.savefig(f'distance_image_histogram_{i}.png')
plt.show()

# Visualize distance image
plt.imshow(distance_image, cmap='jet')  # distance_image
plt.imshow(distance_image, cmap='jet')
plt.title("Distance Image")
plt.colorbar()
# plt.savefig(f'distance_image_{i}.png')
plt.show()

# Apply thresholding to distance image
thresholded_distance_image = np.where(distance_image > 3, 0, distance_image)

# plot thresholded distance image
plt.imshow(thresholded_distance_image, cmap='jet')
plt.title("Thresholded Distance Image")
plt.colorbar()
# plt.savefig(f'thresholded_distance_image_{i}.png')
plt.show()


# Apply mean filter to distance image
mean_filtered_distance_image = cv2.blur(thresholded_distance_image, (5, 5))

# plot the hitogram of filtered distance_image
# plt.hist(mean_filtered_distance_image.flatten(), bins=100)
# plt.title("Mean Filtered Distance Image Histogram")
# plt.xlabel("Distance")
# plt.ylabel("Frequency")
# # plt.savefig(f'mean_filtered_distance_image_histogram_{i}.png')
# plt.show()

# Visualize distance image with mean filter applied
plt.imshow(mean_filtered_distance_image, cmap='jet')
plt.title("Mean Filtered Distance Image")
plt.colorbar()
# plt.savefig(f'mean_filtered_distance_image_{i}.png')
plt.show()


# Apply median filter to distance image
median_filtered_distance_image = medfilt2d(thresholded_distance_image)

# plot the hitogram of filtered distance_image
# plt.hist(median_filtered_distance_image.flatten(), bins=100)
# plt.title("Median Filtered Distance Image Histogram")
# plt.xlabel("Distance")
# plt.ylabel("Frequency")
# plt.savefig(f'median_filtered_distance_image_histogram_{i}.png')
# plt.show()

# Visualize distance image with median filter applied
plt.imshow(median_filtered_distance_image, cmap='jet')
plt.title("Median Filtered Distance Image")
plt.colorbar()
# plt.savefig(f'median_filtered_distance_image_{i}.png')
plt.show()



######### 2.RANSAC #########
# Filter out points with z-component = 0 (invalid measurements)
point_cloud_c = np.reshape(point_cloud, (424 * 512, 3))  # Sequence of 3d points
valid_points = point_cloud_c[point_cloud_c[:, 2] != 0]

# Run class RANSAC to get the best plane of floor and the 1d inliers mask
plane1, inliers_floor = RANSAC().ransac_plane(valid_points, n_iterations=100, threshold=None)  # 0.07, 0.05, 0.1
# print("plane1:",plane1)

# Create binary mask image for the floor
floor_mask_image = np.zeros((424, 512), dtype=int)
valid_indices = np.nonzero(point_cloud[:, :, 2])  # Get valid indices, tuple:(array([]), array([]))
floor_mask_image[valid_indices[0], valid_indices[1]] = inliers_floor.astype(int)

# Plot the mask image
plt.imshow(floor_mask_image, cmap='jet')
plt.title('Floor Mask')
# plt.savefig(f'Floor_Mask_{i}.png')
plt.show()


######### 3.Filtering on the Mask Image #########
orig_mask_image = floor_mask_image.astype(np.uint8)

# Perform Morphological Operators, Opening (erosion followed by dilation)
kernel = np.ones((10, 10), np.uint8)  # Define kernel for erosion/dilation
mask_open = orig_mask_image.copy()
mask_open = cv2.erode(mask_open, kernel, iterations=1)  # Erosion
mask_open = cv2.dilate(mask_open, kernel, iterations=1)  # Dilation

# Plot the mask image
plt.imshow(mask_open, cmap='jet')
plt.title('Opening filtered floor mask')
# plt.savefig(f'Opening_filtered_floor_mask_{i}.png')
plt.show()

# Perform Closing (dilation followed by erosion)
mask_close= orig_mask_image.copy()
mask_close = cv2.dilate(mask_close, kernel, iterations=1)  # Dilation
mask_close = cv2.erode(mask_close, kernel, iterations=1)  # Erosion

# Plot the mask image
plt.imshow(mask_close, cmap='jet')
plt.title('Closing filtered floor mask')
# plt.savefig(f'Closing_filtered_floor_mask_{i}.png')
plt.show()


######### 4.Finding the Top Plane of the Box #########
# Filter out points belonging to the floor
points_not_floor = valid_points[inliers_floor == 0]

# Run the class RANSAC to get the best plane of box and the 1d inliers mask
plane2, inliers_box = RANSAC().ransac_plane(points_not_floor, n_iterations=100, threshold=0.01)  # 0.02
# print("plane2:",plane2)

# Create new binary mask image for the box
box_mask_image = np.zeros((424, 512), dtype=int)
outlier_indices = np.zeros((2, points_not_floor.shape[0]))
outlier_indices[0] = valid_indices[0][inliers_floor == 0]
outlier_indices[1] = valid_indices[1][inliers_floor == 0]
box_mask_image[outlier_indices[0].astype(int), outlier_indices[1].astype(int)] = inliers_box.astype(int)


# Plot the mask image
plt.imshow(box_mask_image, cmap='jet')
plt.title('Box Mask')
plt.savefig(f'Box_Mask_{i}.png')
plt.show()

# Find largest connected component in box mask image
labeled_mask, num_labels = label(box_mask_image)
largest_component = np.argmax(np.bincount(labeled_mask.flatten())[1:]) + 1
largest_component_mask = (labeled_mask == largest_component)

plt.imshow(largest_component_mask, cmap='jet')
plt.title('Largest component mask')
# plt.savefig(f'Largest_component_mask_{i}.png')
plt.show()

# Perform morphological operations on box mask image
kernel = np.ones((3, 3), np.uint8)
mask_box_erosion = cv2.erode(largest_component_mask.astype(np.uint8), kernel, iterations=1)
mask_box_open = cv2.dilate(mask_box_erosion, kernel, iterations=1)

# Plot the mask image
plt.imshow(mask_box_open, cmap='jet')
plt.title('Opening filtered box maskn')
plt.show()

# Perform dilation followed by erosion (closing)
mask_box_dilation = cv2.dilate(largest_component_mask.astype(np.uint8), kernel, iterations=1)  # Dilation
mask_box_close = cv2.erode(mask_box_dilation, kernel, iterations=1)  # Erosion

# Plot the mask image
plt.imshow(mask_box_close, cmap='jet')
plt.title('Closing filtered box mask')
# plt.savefig(f'Closing_filtered_box_mask_{i}.png')
plt.show()


# perform median filtering
mask_box_median = medfilt2d(mask_box_close)
mask_box_mean = cv2.blur(mask_box_close, (3, 3))

# Plot the mask image
plt.imshow(mask_box_median, cmap='jet')
plt.title('Median filtered box mask')
plt.show()

#plot the mean filtered mask
plt.imshow(mask_box_mean, cmap='jet')
plt.title('Mean filtered box mask')
plt.show()


######### 5.Measuring the Dimensions of the Box #########
input_mask = mask_box_close  # Take the processed box mask

# Find the coordinates of box pixels
points = np.column_stack(np.where(input_mask > 0))

# Find the corners of the box
leftmost = np.argmin(points[:,0] + points[:,1])
rightmost = np.argmax(points[:,0] + points[:,1])
# bottommost and topmost have the extreme signed distance to the line of leftmost and rightmost
bottommost = np.argmax(np.cross(points[rightmost] - points[leftmost], points - points[leftmost])
                       /np.linalg.norm(points[rightmost]-points[leftmost]), axis=0)
topmost = np.argmin(np.cross(points[rightmost] - points[leftmost], points - points[leftmost])
                    /np.linalg.norm(points[rightmost]-points[leftmost]), axis=0)

# Extract the x and y coordinates of the corners
top_right = points[bottommost]
bottom_right = points[rightmost]
bottom_left = points[topmost]
top_left = points[leftmost]

# Plot the corners
# plt.scatter(top_right[1], top_right[0], c='r')
# plt.scatter(bottom_right[1], bottom_right[0], c='b')
# plt.scatter(bottom_left[1], bottom_left[0], c='g')
# plt.scatter(top_left[1], top_left[0], c='y')
# plt.imshow(input_mask, cmap='jet')
# plt.title('Box Corners')
# plt.show()

# Calculate the size of the box
width = (np.linalg.norm(point_cloud[tuple(top_right)] - point_cloud[tuple(bottom_right)])
         +np.linalg.norm(point_cloud[tuple(bottom_left)] - point_cloud[tuple(top_left)]))/2
length = (np.linalg.norm(point_cloud[tuple(bottom_right)] - point_cloud[tuple(bottom_left)])
          +np.linalg.norm(point_cloud[tuple(top_left)] - point_cloud[tuple(top_right)]))/2

print("length:", length)
print("width:", width)

# Get the 3d box points
box_points = point_cloud[input_mask > 0]
height = np.mean(np.abs(np.dot(box_points, plane1[:-1]) + plane1[-1]) / np.linalg.norm(plane1[:-1]))

print("height:",height)

# Plot the boundary of the box
fig, ax = plt.subplots()
ax.imshow(input_mask, cmap='jet')
ax.plot([top_left[1], top_right[1]], [top_left[0], top_right[0]], c='y')
ax.plot([top_right[1], bottom_right[1]], [top_right[0], bottom_right[0]], c='y')
ax.plot([bottom_right[1], bottom_left[1]], [bottom_right[0], bottom_left[0]], c='y')
ax.plot([bottom_left[1], top_left[1]], [bottom_left[0], top_left[0]], c='y')
plt.title('Boundary of the box')
plt.savefig(f'Boundary_of_the_box_{i}.png')
plt.show()


