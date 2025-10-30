import numpy as np
from pygame.math import Vector2 as v2
def is_point_between(p: v2, p1: v2, p2: v2, threshold: float = 1e-5) -> bool:
    """
    Check if a point `p` lies between two points `p1` and `p2` within a certain threshold.

    Parameters:
        p (np.ndarray): The point to test, as a NumPy array [x, y].
        p1 (np.ndarray): One endpoint of the segment, as a NumPy array [x, y].
        p2 (np.ndarray): The other endpoint of the segment, as a NumPy array [x, y].
        threshold (float): The tolerance threshold for collinearity and bounds.

    Returns:
        bool: True if `p` lies between `p1` and `p2` within the threshold, otherwise False.
    """

    # Check if `p` is approximately collinear with `p1` and `p2`
    vector_1 = p - p1
    vector_2 = p2 - p1

    # Compute the cross product magnitude to check collinearity (2D case)
    cross_product = abs(np.cross(vector_1, vector_2) / np.linalg.norm(vector_2))
    if cross_product > threshold:
        return False

    # Check if `p` lies within the bounds of the line segment [p1, p2]
    min_x, max_x = sorted([p1[0], p2[0]])
    min_y, max_y = sorted([p1[1], p2[1]])

    if not (min_x - threshold <= p[0] <= max_x + threshold and 
            min_y - threshold <= p[1] <= max_y + threshold):
        return False

    return True

if __name__ == "__main__":
    b = is_point_between(v2([20,20]), v2([0,50]), v2([0,0]), threshold=10)
    print(b)
