import numpy as np
import time
from scipy.linalg import blas

class Test:
    def __init__(self, test_size):
        self.num_elements = test_size
        self.data = np.random.randn(self.num_elements).astype(np.float32)        
    
    def reduce(self):
         blas.sasum(self.data) 

for size in range(0, 9):
    total_time = 0.0
    for _ in range(0,4):
        test = Test(2**size * 1024 * 1024)
        start = time.time()
        test.reduce()
        total_time += time.time() - start 
    print(f"Time taken to reduce is {8 * 2**size /1024} GiB with multi threaded CPU is {total_time/4}")