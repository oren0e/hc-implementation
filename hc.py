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



