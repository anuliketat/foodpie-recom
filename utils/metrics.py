import numpy as np
from numpy.linalg import norm


def cosine_sim(a, b):
	return np.dot(a, b)/(norm(a)*norm(b))

def eucledian_dist(a, b):
	return norm(a-b)
