import math
from collections import defaultdict

def get_midpoint(points):
    """
    Calculate the midpoint(center point) between two points

    Args:
        points: [x1, y1, x2, y2]

    Return:
        tuple: (int: x, int:y)
    """
    assert len(points) == 4 # array should be of length 4

    x1, y1, x2, y2 = points

    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    return (mid_x, mid_y)

import math
from collections import defaultdict

def cluster_points(points, threshold):
    """
    Cluster points and return both coordinates and original indices
    
    Args:
        points: List of coordinate tuples/lists
        threshold: Maximum allowed distance between connected points
        
    Returns:
        tuple: (list of point clusters, list of index clusters)
    """
    if not points:
        return [], []
    if threshold < 0:
        raise ValueError("Threshold must be non-negative")
    
    # Validate dimensions
    dim = len(points[0])
    if any(len(p) != dim for p in points):
        raise ValueError("All points must have the same dimensions")
    
    n = len(points)
    parent = list(range(n))
    
    def find(x):
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        parent[find(y)] = find(x)
    
    # Cluster using math.dist with overflow protection
    for i in range(n):
        for j in range(i+1, n):
            try:
                if math.dist(points[i], points[j]) <= threshold:
                    union(i, j)
            except OverflowError:
                continue
    
    # Create both point clusters and index clusters
    point_clusters = defaultdict(list)
    index_clusters = defaultdict(list)
    
    for idx in range(n):
        root = find(idx)
        point_clusters[root].append(points[idx])
        index_clusters[root].append(idx)
    
    return (
        list(point_clusters.values()), 
        list(index_clusters.values())
    )
