import numpy as np

class PlaneEstimator:
    def estimate_plane(self, points):

        """
        Determines the coefficients of the plane equation (ax + by + cz + d = 0) given an array of three points in 3D space.

        Args:
            points (np.array): A numpy array of shape (3,3) representing three points in 3D space. Each row contains the x, y, and z coordinates of a point, respectively.

        Returns:
            tuple: A tuple containing the coefficients (a, b, c, d) of the plane equation.
        """
        # Extracting the coordinates of the three points
        x1, y1, z1 = points[0]
        x2, y2, z2 = points[1]
        x3, y3, z3 = points[2]

        # Calculating the vectors between the points
        vector1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        vector2 = np.array([x3 - x1, y3 - y1, z3 - z1])

        # Calculating the normal vector of the plane using the cross product of the two vectors
        normal_vector = np.cross(vector1, vector2)

        # Extracting the coefficients (a, b, c) of the plane equation
        a, b, c = normal_vector

        # Calculating the constant term (d) of the plane equation using the dot product of the normal vector and one of the points
        d = -np.dot(normal_vector, points[0])

        return [a, b, c, d]



    def ransac_plane(self, points, n_iterations=100, threshold=0.1):
        """
        Estimate a plane from a set of 3D points using RANSAC (Random Sample Consensus).
        """
        best_plane = None
        best_error = float('inf')
        inliers_mask = np.zeros(points.shape[0], dtype=bool)
        for i in range(n_iterations):
            # Choose three points at random (3, 3)
            samples = points[np.random.choice(points.shape[0], 3, replace=False)]

            # Estimate a plane using these points
            plane = self.estimate_plane(samples)

            # Compute the distance between the plane and all points
            errors = np.abs(np.dot(points, plane[:-1]) + plane[-1])

            # Count the number of inliers (points within the threshold distance of the plane)
            inliers = errors < threshold

            # Update the best plane if this one is better
            if np.sum(inliers) > np.sum(inliers_mask) and np.sum(inliers) > 0 and np.sum(errors) < best_error:
                best_plane = plane
                best_error = np.sum(errors)
                inliers_mask = inliers

        # Return the best plane and the inliers mask
        return best_plane, inliers_mask

