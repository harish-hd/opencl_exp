import pyopencl as cl
import numpy as np
import os
import time
os.environ['PYOPENCL_CTX']='0'

class Test:
    def __init__(self, test_size):
        self.num_elements = test_size
        self.data = np.random.randn(self.num_elements).astype(np.float32)   

    @classmethod
    def setup_kernel(cls, context): 
        prg = cl.Program(context, """
        __kernel void reduce(ulong n, __global float *input, __global float *out)
        {
        float sum = 0.0;
        for(int k=0; k<n; k++)
        {
            sum += *input++;
        }
        *out = sum;
        }
        """).build()
        return prg

    def reduce(self):
        buffer_size = self.data.nbytes 
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)    
        prg = Test.setup_kernel(context)
        reduce_result = np.zeros(1).astype(np.float32)
        mf = cl.mem_flags
        device_buffer = cl.Buffer(context, flags = mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.data)
        output_buffer = cl.Buffer(context, mf.WRITE_ONLY, reduce_result.nbytes)
        prg.reduce(queue, reduce_result.shape , None, np.uint64(self.num_elements), device_buffer, output_buffer)
        device_buffer.release()
        cl.enqueue_copy(queue, reduce_result, output_buffer)
         

for size in range(0, 9):
    total_time = 0.0
    for _ in range(0,4):
        test = Test(2**size * 1024 * 1024)
        start = time.time()
        test.reduce()
        total_time += time.time() - start 
    print(f"Time taken to reduce is {8 * 2**size /1024} GiB with single threaded GPU is {total_time/4}")