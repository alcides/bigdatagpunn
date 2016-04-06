import copy

import numpy as np
import pandas as pd
import numba
import math
import time
from numba import cuda
df = pd.read_csv("datasets/wine.txt", sep="\t", header=None)

instances = df.shape[0]
train_instances = 20
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

@cuda.jit(device=True, inline=True)
def nonlin_g(x):
	return 1/(1+math.exp(-x))

@cuda.jit(device=True, inline=True)
def nonlind_g(x):
	return x*(1-x)
    
    
@cuda.jit(argtypes=[numba.float32[:,:], numba.float32[:,:], numba.float32[:,:], numba.float32[:,:], numba.int32])
def train_kernel(X, y, syn0, syn1, iterations):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    j = tx + bx * bw
    i = ty + by * bh
    syn1[i,j] = 42.0
    syn0[i,j] = 42.0
    """
    instances = train_instances
    l1 = cuda.shared.array(shape=(instances, ndims), dtype=numba.float64)
    l2_delta = cuda.shared.array(shape=(instances, 3), dtype=numba.float64)
    l1_delta = cuda.shared.array(shape=(instances, ndims), dtype=numba.float64)
    i, j = cuda.grid(2)
    for it in range(iterations):
        if i < instances and j < ndims:
            acc = 0
            for k in range(ndims):
                acc += X[i, k] * syn0[k, j]
            l1[i, j] = nonlin_g(acc)
            cuda.syncthreads()
            if j < 3:
                acc = 0
                for k in range(ndims):
                    acc += l1[i,k] * syn1[k,j]
                l2 = nonlin_g(acc)
                l2_error = y[i, j] - l2
                l2_delta[i, j] = l2_error * nonlind_g(l2)
            cuda.syncthreads()
            acc = 0
            for k in range(ndims):
                acc += l2_delta[i,k] * syn1[j, k]
            l1_error = acc
            l1_delta[i,j] = l1_error * nonlind_g(l1[k, j])
            cuda.syncthreads()
            if j < 3:
                acc = 0
                for k in range(ndims):
                    acc += l1[k, i] * l2_delta[k, j]
                syn1[i, j] = 10 #acc
            acc = 0
            for k in range(ndims):
                acc += X[k, i] * l1_delta[k, j]
            syn0[i, j] += acc
            cuda.syncthreads()
            """
def train_cuda(X, y, conf, iterations=6000):
    stream = cuda.stream()
    with stream.auto_synchronize():
        gpu = cuda.get_current_device()
    
        syn0, syn1 = conf
        syn0g = cuda.to_device(syn0, stream)
        syn1g = cuda.to_device(syn1, stream)
        Xg = cuda.to_device(X, stream)
        yg = cuda.to_device(y, stream)
        rows = X.shape[0]
        thread_ct = (gpu.WARP_SIZE, gpu.WARP_SIZE)
        block_ct = map(lambda x: int(math.ceil(float(x) / gpu.WARP_SIZE)), [rows, ndims])
        train_kernel[block_ct, thread_ct, stream](Xg, yg, syn0g, syn1g, iterations)
        syn0g.to_host(stream)
        syn1g.to_host(stream)
        print syn1
        print syn0
        return (syn0, syn1)

def config_error(df, conf):
    syn0, syn1 = conf
    X = df.iloc[:,0:ndims].as_matrix()
    y = df.iloc[:,ndims:].as_matrix()
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    # how much did we miss the target value?
    l2_error = y - l2
    return np.mean(np.abs(l2_error))
    
def op_configs(conf, conf2, fun):
    a =  [fun(a,b) for a,b in zip(conf, conf2)]
    return a
 
if __name__ == '__main__':
    iterations = 1
    conf = generate_random_config()
    X = df.iloc[0:train_instances,0:ndims].as_matrix()
    y = df.iloc[0:train_instances,ndims:].as_matrix()
       
    for train_fun in [train_cuda, train]:
        conf_ = copy.deepcopy(conf)
        start = time.time()
        output_conf = train_fun(X, y, conf_, iterations)
        end = time.time()
        #print output_conf[0]
        #print output_conf[1]
        print "Error: ", config_error(df, output_conf), "in", (end-start)