//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for OpenCL
//
//===----------------------------------------------------------------------===//

__kernel void matmul_kernel(
  __global float* var_0,
  __global float* var_1,
  __global float* var_2,
  int var_3,
  int var_4,
  int var_5,
  int var_6,
  int var_7,
  int var_8
) {	// L5
  event_t ev = 0;
  __local float var_9[16][16];	// L32
  __local float var_10[16][16];	//
  for (int var_11 = 0; var_11 < 16; var_11 += 1) {	//
    for (int var_12 = 0; var_12 < 16; var_12 += 1) {	//
      var_10[var_11][var_12] = 0.000000;	//
    }
  }
  int var_13 = get_global_id(0);
  int var_15 = get_global_id(1);
  int var_17 = var_15 * 16;	// L16
  int var_18 = var_13 * 16;	// L17
  int var_21 = var_17 * var_6;	// L20
  __local float var_23[16][1];	// L30
  __local float var_24[1][16];	// L31
  __local float var_25[16][16];	// L32
  for (int var_26 = 0; var_26 < var_5; var_26 += 1) {	// L26
    int var_28 = var_21 + var_26;	// L27
    int var_29 = var_26 * var_7;	// L28
    int var_31 = var_29 + var_18;	// L28
    for (int i = 0; i < 16; i += 1) {
      ev = async_work_group_copy(var_23 + i, var_0 + var_28 + i * var_6, 1, 0);
    }	// L30
    wait_group_events(1, &ev);
    ev = async_work_group_copy(var_24, var_1 + var_31, 16, 0);	// L31
    wait_group_events(1, &ev);
    for (int var_32 = 0; var_32 < 16; var_32 += 1) {	// L32
      for (int var_33 = 0; var_33 < 16; var_33 += 1) {	// L32
        float var_34 = var_23[var_32][0];	// L32
        var_25[var_32][var_33] = var_34;	// L32
      }
    }
    for (int var_35 = 0; var_35 < 16; var_35 += 1) {	// L32
      for (int var_36 = 0; var_36 < 16; var_36 += 1) {	// L32
        float var_37 = var_24[0][var_36];	// L32
        var_9[var_35][var_36] = var_37;	// L32
      }
    }
    for (int var_38 = 0; var_38 < 16; var_38 += 1) {	// L32
      for (int var_39 = 0; var_39 < 16; var_39 += 1) {	// L32
        float var_40 = var_25[var_38][var_39];	// L32
        float var_41 = var_9[var_38][var_39];	// L32
        float var_42 = var_40 * var_41;	// L32
        var_25[var_38][var_39] = var_42;	// L32
      }
    }
    for (int var_43 = 0; var_43 < 16; var_43 += 1) {	// L32
      for (int var_44 = 0; var_44 < 16; var_44 += 1) {	// L32
        float var_45 = var_10[var_43][var_44];	// L32
        float var_46 = var_25[var_43][var_44];	// L32
        float var_47 = var_45 + var_46;	// L32
        var_10[var_43][var_44] = var_47;	// L32
      }
    }
  }
  int var_49 = var_17 * var_8;	// L35
  int var_50 = var_49 + var_18;	// L35
  int var_51 = var_17 + 16;	// L39
  int var_53 = min(var_51, var_3);	// L39
  int var_54 = max(var_53, var_17);	// L39
  int var_55 = var_54 - var_17;	// L39
  int var_56 = var_18 + 16;	// L39
  int var_58 = min(var_56, var_4);	// L39
  int var_59 = max(var_58, var_18);	// L39
  int var_60 = var_59 - var_18;	// L39
  int var_61 = min(var_55, 16);	// L39
  int var_62 = min(var_60, 16);	// L39
  for (int i = 0; i < var_61; i += 1) {
    ev = async_work_group_copy(var_2 + var_50 + i * var_8, var_10 + i, var_62, 0);
  }	// L39
  wait_group_events(1, &ev);
}

