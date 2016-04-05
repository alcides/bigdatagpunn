import sys

from nn import *
from queue import *

def worker(wid):
    metadata = r.blpop('worker_%d' % wid)[1]
    data = r.blpop('worker_data_%d' % wid)[1]
    data2 = r.blpop('worker_data2_%d' % wid)[1]
    
    a, b, iterations, conf = decode_req(metadata, data, data2)
    
    X = df.iloc[a:b,0:13].as_matrix()
    y = df.iloc[a:b,13:].as_matrix()
    output_conf = train(X, y, conf, iterations)
    
    metadata, data, data2 = encode_req(a, b, iterations, output_conf)
    
    r.rpush("master_%d" % wid, metadata)
    r.rpush("master_data_%d" % wid, data)
    r.rpush("master_data2_%d" % wid, data2)

if __name__ == '__main__':
    wid = int(sys.argv[1])
    while True:
        worker(wid)