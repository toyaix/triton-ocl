import ctypes
import numpy as np

# Load OpenCL shared library
cl = ctypes.CDLL("libOpenCL.so")  # Use "OpenCL.dll" on Windows

# Constant definitions
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
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

def launch(gridX, gridY, gridZ, tt_kernel, bound_args):
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
    program_src_ptr = ctypes.c_char_p(tt_kernel.kernel)
    program_size = ctypes.c_size_t(len(tt_kernel.kernel))
    program = cl.clCreateProgramWithSource(context, 1, ctypes.byref(program_src_ptr), ctypes.byref(program_size), ctypes.byref(err))
    build_program = cl.clBuildProgram(program, 1, devices, None, None, None)
    if build_program != CL_SUCCESS:
        print('clBuildProgram', build_program)
        exit(0)

    # Create kernel and set arguments
    kernel = cl.clCreateKernel(program, tt_kernel.name.encode(), ctypes.byref(err))
    args_lst = list(bound_args)
    a, b, c, n = args_lst[0], args_lst[1], args_lst[2], args_lst[3]
    a_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, a.element_size() * a.numel(), ctypes.c_void_p(a.data_ptr()), ctypes.byref(err)))
    b_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, b.element_size() * b.numel(), ctypes.c_void_p(b.data_ptr()), ctypes.byref(err)))
    c_buf = ctypes.c_void_p(cl.clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, c.element_size() * c.numel(), ctypes.c_void_p(c.data_ptr()), ctypes.byref(err)))
    cl.clSetKernelArg(kernel, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(a_buf))
    cl.clSetKernelArg(kernel, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(b_buf))
    cl.clSetKernelArg(kernel, 2, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(c_buf))
    int_value = ctypes.c_int(n)
    cl.clSetKernelArg(kernel, 3, ctypes.sizeof(int_value), ctypes.byref(int_value))

    # Launch kernel
    global_size = (ctypes.c_size_t * 3)(gridX, gridY, gridZ)
    local_size = (ctypes.c_size_t * 3)(1, 1, 1)
    launch_kenrel = cl.clEnqueueNDRangeKernel(queue, kernel, 3, None, global_size, local_size, 0, None, None)
    if launch_kenrel != CL_SUCCESS:
        print('clEnqueueNDRangeKernel', launch_kenrel)
    cl.clFinish(queue)

    # Read results from device
    cl.clEnqueueReadBuffer(queue, a_buf, False, 0, a.element_size() * a.numel(), ctypes.c_void_p(a.data_ptr()), 0, None, None)
    cl.clEnqueueReadBuffer(queue, b_buf, False, 0, b.element_size() * b.numel(), ctypes.c_void_p(b.data_ptr()), 0, None, None)
    cl.clEnqueueReadBuffer(queue, c_buf, False, 0, c.element_size() * c.numel(), ctypes.c_void_p(c.data_ptr()), 0, None, None)
    cl.clFinish(queue)
