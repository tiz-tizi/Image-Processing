import numpy as np

class RANSAC:
    def estimate_plane(self, points):

        """
        Determines the coefficients of the plane equation (ax + by + cz + d = 0) given an array of three points in 3D space.
        """
        x1, y1, z1 = points[0]
        x2, y2, z2 = points[1]
        x3, y3, z3 = points[2]

        # Calculating the vectors between the points
        vector1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        vector2 = np.array([x3 - x1, y3 - y1, z3 - z1])

        # Calculating the normal vector of the plane
        normal_vector = np.cross(vector1, vector2)
        a, b, c = normal_vector

        # Calculating the constant term
        d = -np.dot(normal_vector, points[0])

        return [a, b, c, d]



    def ransac_plane(self, points, n_iterations=100, threshold=0.01):
        """
        Estimate a plane from a set of 3D points using RANSAC (RANdom Sampling And Consensus).
        """
        best_plane = None
        inliers_mask = np.zeros(points.shape[0], dtype=bool)  # 1d mask for the inliers

        auto = False
        if threshold is None:
            auto = True  # Automatically determine the threshold

        distance_list = []
        for i in range(n_iterations):
            # Choose three points at random
            samples = points[np.random.choice(points.shape[0], 3, replace=False)]
            plane = self.estimate_plane(samples)

            # Compute the distance between the plane and all points
            distance = np.abs(np.dot(points, plane[:-1]) + plane[-1]) / np.linalg.norm(plane[:-1])

            # Automatically determine the threshold using the median
            if auto:
                distance_list.append(distance)
                threshold = np.median(distance_list)

            # Get the inliers
            inliers = distance < threshold

            # Update the best plane
            if np.sum(inliers) > np.sum(inliers_mask) and np.sum(inliers) > 0:
                best_plane = plane
                inliers_mask = inliers
        print("threshold:", threshold)
        # Return the best plane and the inliers mask
        return best_plane, inliers_mask

