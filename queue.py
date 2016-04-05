import numpy as np
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

def encode_req(a,b,it,conf):
    syn0, syn1 = conf
    metadata = "|".join(map(str,[a,b,it, syn0.shape[0], syn0.shape[1], syn0.dtype, syn1.shape[0], syn1.shape[1], syn1.dtype ]))
    data = conf[0].ravel().tostring()
    data2 = conf[1].ravel().tostring()
    return metadata, data, data2
    
    
def decode_req(metadata, data, data2):
    a, b, iterations, l, w, array_dtype, l2, w2, array_dtype2 = metadata.split('|')
    syn0 = np.fromstring(data, dtype=array_dtype)
    syn0 = syn0.reshape(int(l), int(w))
    syn1 = np.fromstring(data2, dtype=array_dtype2).reshape(int(l2), int(w2))
    conf = (syn0, syn1)
    return int(a), int(b), int(iterations), conf