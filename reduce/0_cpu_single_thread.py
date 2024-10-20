import numpy as np
import time

class Test:
    def __init__(self, test_size):
        self.num_elements = test_size
        self.data = np.random.randn(self.num_elements).astype(np.float32)        
    
    def reduce(self):
        np.sum(self.data) # numpy.sum is apparently single threaded

for size in range(0, 9):
    total_time = 0.0
    for _ in range(0,4):
        test = Test(2**size * 1024 * 1024)
        start = time.time()
        test.reduce()
        total_time += time.time() - start 
    print(f"Time taken to reduce is {8 * 2**size /1024} GiB with single threaded CPU is {total_time/4}")