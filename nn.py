import numpy as np
import pandas as pd
import copy
df = pd.read_csv("datasets/wine.txt", sep="\t", header=None)

instances = df.shape[0]
ndims = 13
nclasses = 3

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

np.random.seed(1)

def generate_random_config():
    syn0 = 2 * np.random.random((ndims, ndims)) - 1
    syn1 = 2 * np.random.random((ndims, nclasses)) - 1
    return (syn0, syn1)

def train(X, y, conf, iterations=6000):
    syn0, syn1 = conf
    for j in xrange(iterations):
    	# Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        # how much did we miss the target value?
        l2_error = y - l2
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
    return (syn0, syn1)
    
def config_error(df, conf):
    syn0, syn1 = conf
    X = df.iloc[:,0:13].as_matrix()
    y = df.iloc[:,13:].as_matrix()
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    # how much did we miss the target value?
    l2_error = y - l2
    
    return np.mean(np.abs(l2_error))
    
def op_configs(conf, conf2, fun):
    a =  [fun(a,b) for a,b in zip(conf, conf2)]
    return a