import ctypes
import numpy as np

# Load OpenCL shared library
cl = ctypes.CDLL("libOpenCL.so")  # Use "OpenCL.dll" on Windows

# Constant definitions
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_WRITE = 1 << 0
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_SUCCESS = 0

# Define argument types for used OpenCL functions
cl.clGetPlatformIDs.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
cl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
cl.clCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_int)]
cl.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
cl.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int)]
cl.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
cl.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
cl.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
cl.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint,
                                      ctypes.c_void_p, ctypes.c_void_p]
cl.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                   ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
cl.clFinish.argtypes = [ctypes.c_void_p]

# Create platform, device, context, and command queue
num_platforms = ctypes.c_uint()
cl.clGetPlatformIDs(0, None, ctypes.byref(num_platforms))
platforms = (ctypes.c_void_p * num_platforms.value)()
cl.clGetPlatformIDs(num_platforms.value, platforms, None)

num_devices = ctypes.c_uint()
cl.clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(num_devices))
devices = (ctypes.c_void_p * num_devices.value)()
cl.clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, None)

err = ctypes.c_int()
context = cl.clCreateContext(None, 1, devices, None, None, ctypes.byref(err))
queue = cl.clCreateCommandQueue(context, devices[0], 0, ctypes.byref(err))

# Create host data
n = 16231
a_np = np.random.rand(n).astype(np.float32)
b_np = np.random.rand(n).astype(np.float32)
c_np = np.empty_like(a_np)

# Create OpenCL buffers
a_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a_np.nbytes, a_np.ctypes.data_as(ctypes.c_void_p), ctypes.byref(err)))
b_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b_np.nbytes, b_np.ctypes.data_as(ctypes.c_void_p), ctypes.byref(err)))
c_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c_np.nbytes, c_np.ctypes.data_as(ctypes.c_void_p), ctypes.byref(err)))

# Define OpenCL kernel source
program_src = b"""
__kernel void add_kernel(
  __global float* var_0,
  __global float* var_1,
  __global float* var_2,
  int var_3
) {	// L2
  int var_4 = get_global_id(0);
  int var_6 = var_4 * 1024;	// L7
  __local float var_8[1024];	// L10
  int var_9 = var_6 + 1024;	// L11
  int var_11 = min(var_9, var_3);	// L13
  int var_12 = max(var_11, var_6);	// L14
  int var_13 = var_12 - var_6;	// L15
  event_t copy_event = async_work_group_copy(var_8, var_0 + var_6, var_13, 0);
  wait_group_events(1, &copy_event);
  __local float var_14[1024];	// L20
  event_t copy_event1 = async_work_group_copy(var_14, var_1 + var_6, var_13, 0);
  wait_group_events(1, &copy_event1);
  for (int var_15 = 0; var_15 < 1024; var_15 += 1) {	// L24
    float var_16 = var_8[var_15];	// L25
    float var_17 = var_14[var_15];	// L26
    float var_18 = var_16 + var_17;	// L27
    var_8[var_15] = var_18;	// L28
  }
  for (int i = 0; i < var_13; i += 1) {
    var_2[i + var_6] = var_8[i];
  }	// L33
}
"""

program_src_ptr = ctypes.c_char_p(program_src)
program_size = ctypes.c_size_t(len(program_src))
program = cl.clCreateProgramWithSource(context, 1, ctypes.byref(program_src_ptr), ctypes.byref(program_size), ctypes.byref(err))
build_program = cl.clBuildProgram(program, 1, devices, None, None, None)
if build_program != CL_SUCCESS:
    print('clBuildProgram', build_program)
    exit(0)

# Create kernel and set arguments
kernel = cl.clCreateKernel(program, b"add_kernel", ctypes.byref(err))
cl.clSetKernelArg(kernel, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(a_buf))
cl.clSetKernelArg(kernel, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(b_buf))
cl.clSetKernelArg(kernel, 2, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(c_buf))
int_value = ctypes.c_int(n)
err = cl.clSetKernelArg(kernel, 3, ctypes.sizeof(int_value), ctypes.byref(int_value))

block_size = 1024
num_sm = (n + block_size - 1) // block_size

# Launch kernel
global_size = (ctypes.c_size_t * 3)(num_sm * block_size, 1, 1)
local_size = (ctypes.c_size_t * 3)(1, 1, 1)
launch_kenrel = cl.clEnqueueNDRangeKernel(queue, kernel, 3, None, global_size, local_size, 0, None, None)
if launch_kenrel != CL_SUCCESS:
    print('clEnqueueNDRangeKernel', launch_kenrel)
cl.clFinish(queue)

# Read results from device
cl.clEnqueueReadBuffer(queue, a_buf, False, 0, a_np.nbytes, a_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)
cl.clEnqueueReadBuffer(queue, b_buf, False, 0, b_np.nbytes, b_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)
cl.clEnqueueReadBuffer(queue, c_buf, False, 0, c_np.nbytes, c_np.ctypes.data_as(ctypes.c_void_p), 0, None, None)
cl.clFinish(queue)

# Verify results
print("Is result correct:", np.allclose(c_np, a_np + b_np))
print("First 10 results:")
print(c_np[:10])
