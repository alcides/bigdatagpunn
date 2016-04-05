import copy

from nn import *
from queue import *

workers = 2
    
def master():
    master_conf = generate_random_config()
    blocks_per_worker = instances/(workers+2)
    
    for k in range(10):
        for i in range(workers):
    
            a = blocks_per_worker * i
            b = blocks_per_worker * (i+1)
            
            print "Scheduling to worker", i, " data from ", a, " to ", b
            metadata, data, data2 = encode_req(a, b, 60000, master_conf)
            r.rpush("worker_%d" % i, metadata)
            r.rpush("worker_data_%d" % i, data)
            r.rpush("worker_data2_%d" % i, data2)
        
        new_conf = copy.deepcopy(master_conf)
        for i in range(workers):
            metadata = r.blpop('master_%d' % i)[1]
            data = r.blpop('master_data_%d' % i)[1]
            data2 = r.blpop('master_data2_%d' % i)[1]
            a, b, iterations, conf = decode_req(metadata, data, data2)
            diff = op_configs(master_conf, conf, lambda a,b: a-b)
            new_conf = op_configs(new_conf, diff, lambda a,b: a+b)

            print "Data from worker", i, "had error:", config_error(df, conf)
            print "Data from worker", i, " merged had error:", config_error(df, new_conf)
        
        master_conf = copy.deepcopy(new_conf)
        
if __name__ == '__main__':
    master()