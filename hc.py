from typing import List, Tuple, NamedTuple, Callable, Union

from math import sqrt

Vector = List[float]

# Define Euclidean distance
def subtract(v: Vector, w: Vector) -> Vector:
    # make sure the vectors are of the same length
    assert len(v) == len(w), "vectors must be of the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

def dot(v: Vector, w: Vector) -> float:
    # make sure the vectors are of the same length
    assert len(v) == len(w), "vectors must be of the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_squares(v: Vector) -> float:
    return dot(v, v)

def distance(v: Vector, w: Vector) -> float:
    # make sure the vectors are of the same length
    assert len(v) == len(w), "vectors must be of the same length"

    return sqrt(sum_squares(subtract(v, w)))


"""
We will represent leaves as NamedTuples. There will be either leaf or merged leaves. A cluster will be either of those.
We will represent the merging hierarchy (order) with the number of clusters left,
so a lower order means upper in the hierarchy merge. 
"""


# Representations
class Leaf(NamedTuple):
    value: Vector


class Merged(NamedTuple):
    children: Tuple
    order: int


Cluster = Union[Leaf, Merged]


# Secondary functions
def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value for child in cluster.children for value in get_values(child)]


def cluster_distance(cluster1: Cluster, cluster2: Cluster, distance_agg: Callable = max) -> float:
    return distance_agg([distance(v1, v2)]
                        for v1 in get_values(cluster1)
                        for v2 in get_values(cluster2))


def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float("inf")     # was never merged
    else:
        return cluster.order


def get_children(cluster: Cluster) -> Tuple:
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children


# Clustering function
def bottom_up_cluster(inputs: List[Vector], distance_agg: Callable = max) -> Cluster:
    # start with all leaves as clusters
    clusters: List[Cluster] = [Leaf(input) for input in inputs]

    # helper function
    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)

    # while we have more than 1 clusters left...
    while len(clusters) > 1:
        # take the two closest clusters
        c1, c2 = min(((cluster1, cluster2) for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]), key=pair_distance)

        # remove them from the list
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them using merge_order = # clusters left
        merged_cluster = Merged((c1, c2), order=len(clusters))

        # and add their merge back to the list
        clusters.append(merged_cluster)

    # once we have only 1 clusters left
    return clusters[0]


# Cluster construction function
def generate_clusters(base_cluster: Cluster, num_clusters: int) -> List[Cluster]:
    # start with the base cluster
    clusters = [base_cluster]

    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # take the latest merge
        next_cluster = min(clusters, key=get_merge_order)

        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]

        # and add back its children (unmerge it)
        clusters.extend(get_children(next_cluster))

    # once we have enough clusters, return it
    return clusters


# Usage testing
cls = bottom_up_cluster([[13,2,-1],[99,100,4],[15,-16,17]])

generate_clusters(cls, num_clusters=3)
generate_clusters(cls, num_clusters=2)
generate_clusters(cls, num_clusters=1)
generate_clusters(cls, num_clusters=4)  # should raise an error!
