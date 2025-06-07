//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for OpenCL
//
//===----------------------------------------------------------------------===//

__kernel void add_kernel(
  __global float* var_0,
  __global float* var_1,
  __global float* var_2,
  int var_3
) {	// L29
  int var_4 = get_global_id(0);
  int var_6 = var_4 * 1024;	// L43
  __local float var_8[1024];	// L49
  int var_9 = var_6 + 1024;	// L49
  int var_11 = min(var_9, var_3);	// L49
  int var_12 = max(var_11, var_6);	// L49
  int var_13 = var_12 - var_6;	// L49
  for (int i = 0; i < var_13; i += 1) {
    var_8[i] = var_0[i + var_6];
  }	// L49
  barrier(CLK_LOCAL_MEM_FENCE);
  __local float var_14[1024];	// L50
  for (int i = 0; i < var_13; i += 1) {
    var_14[i] = var_1[i + var_6];
  }	// L50
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int var_15 = 0; var_15 < 1024; var_15 += 1) {	// L51
    float var_16 = var_8[var_15];	// L51
    float var_17 = var_14[var_15];	// L51
    float var_18 = var_16 + var_17;	// L51
    var_8[var_15] = var_18;	// L51
  }
  for (int i = 0; i < var_13; i += 1) {
    var_2[i + var_6] = var_8[i];
  }	// L53
}

