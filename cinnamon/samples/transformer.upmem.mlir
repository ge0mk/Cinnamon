#map = affine_map<(d0, d1, d2) -> (d1 * 8 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (0)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 8 + d2, 0)>
#map3 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2, 0)>
#map5 = affine_map<(d0, d1, d2) -> (d2)>
module {
  memref.global "private" constant @__constant_1xi64_3 : memref<1xi64> = dense<32768> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256x1xf32 : memref<256x1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_4 : memref<2xi64> = dense<[768, 1]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_2 : memref<1xi64> = dense<768> {alignment = 64 : i64}
  memref.global "private" constant @__constant_48x6xf32 : memref<48x6xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_3 : memref<2xi64> = dense<[48, 6]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_48x1xf32 : memref<48x1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_2 : memref<2xi64> = dense<[288, 1]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_1 : memref<1xi64> = dense<288> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x18xf32 : memref<16x18xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_1 : memref<2xi64> = dense<[16, 18]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_288xf32 : memref<288xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64_0 : memref<1xi64> = dense<48> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x6xf32 : memref<8x6xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64_0 : memref<2xi64> = dense<[8, 6]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_48xf32 : memref<48xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32_0 : memref<f32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<256> {alignment = 64 : i64}
  memref.global "private" constant @__constant_128x2xf32 : memref<128x2xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[128, 2]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_256xf32 : memref<256xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<0xFF800000> {alignment = 64 : i64}
  func.func @forward(%arg0: index, %arg1: index, %arg2: memref<6x256x288xf32>, %arg3: memref<6x256x288xf32>, %arg4: memref<32000x288xf32>, %arg5: memref<6x288xf32>, %arg6: memref<6x288x288xf32>, %arg7: memref<6x288x288xf32>, %arg8: memref<6x288x288xf32>, %arg9: memref<6x288x288xf32>, %arg10: memref<6x768x288xf32>, %arg11: memref<6x288x768xf32>, %arg12: memref<6x768x288xf32>, %arg13: memref<6x288xf32>, %arg14: memref<288xf32>, %arg15: memref<32000x288xf32>) -> memref<32000xf32> {
    %c256 = arith.constant 256 : index
    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %c288 = arith.constant 288 : index
    %c768 = arith.constant 768 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 4.800000e+01 : f32
    %cst_2 = arith.constant 1.000000e+04 : f32
    %subview = memref.subview %arg4[%arg0, 0] [1, 288] [1, 1] : memref<32000x288xf32> to memref<288xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    memref.copy %subview, %alloc : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<256x288xf32>
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<256x288xf32>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<48x6xf32>
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<768x1xf32>
    %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<768x1xf32>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<768x1xf32>
    %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<768x1xf32>
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
    %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
    %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<48x1xf32>
    %0 = scf.for %arg16 = %c0 to %c6 step %c1 iter_args(%arg17 = %alloc) -> (memref<288xf32>) {
      %subview_52 = memref.subview %arg5[%arg16, 0] [1, 288] [1, 1] : memref<6x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %arg17, %alloc_3 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_52, %alloc_4 : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
      %5 = func.call @rmsnorm(%alloc_3, %alloc_4) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
      %subview_53 = memref.subview %arg6[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %subview_54 = memref.subview %arg7[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %subview_55 = memref.subview %arg8[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %6 = memref.get_global @__constant_2xi64_2 : memref<2xi64>
      %reshape_56 = memref.reshape %5(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_5, %alloc_6 : memref<288x1xf32> to memref<288x1xf32>
      %7 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_6) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_53[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_7[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_7[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_8[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_8, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %8 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
      %reshape_57 = memref.reshape %7(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      memref.copy %alloc_9, %alloc_10 : memref<288x1xf32> to memref<288x1xf32>
      %9 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_10) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_54[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_11[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_11[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_12[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_12, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_58 = memref.reshape %9(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      memref.copy %alloc_13, %alloc_14 : memref<288x1xf32> to memref<288x1xf32>
      %10 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_14) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_55[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_56[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_15[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_15[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_16[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_16, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_59 = memref.reshape %10(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %11 = arith.index_cast %arg1 : index to i64
      %12 = arith.uitofp %11 : i64 to f32
      memref.copy %reshape_57, %alloc_17 : memref<288xf32> to memref<288xf32>
      memref.copy %reshape_58, %alloc_18 : memref<288xf32> to memref<288xf32>
      %13:2 = scf.for %arg18 = %c0 to %c288 step %c2 iter_args(%arg19 = %alloc_17, %arg20 = %alloc_18) -> (memref<288xf32>, memref<288xf32>) {
        %27 = arith.remui %arg18, %c48 : index
        %28 = arith.index_cast %27 : index to i64
        %29 = arith.uitofp %28 : i64 to f32
        %30 = arith.divf %29, %cst_1 : f32
        %31 = llvm.intr.pow(%cst_2, %30)  : (f32, f32) -> f32
        %32 = arith.divf %cst_0, %31 : f32
        %33 = arith.mulf %12, %32 : f32
        %34 = llvm.intr.cos(%33)  : (f32) -> f32
        %35 = llvm.intr.sin(%33)  : (f32) -> f32
        %alloc_82 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
        memref.copy %arg19, %alloc_82 : memref<288xf32> to memref<288xf32>
        %36 = func.call @rot(%alloc_82, %arg18, %34, %35) : (memref<288xf32>, index, f32, f32) -> memref<288xf32>
        %37 = arith.cmpi ult, %arg18, %c288 : index
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
        %38 = scf.if %37 -> (memref<288xf32>) {
          memref.copy %arg20, %alloc_83 : memref<288xf32> to memref<288xf32>
          %39 = func.call @rot(%alloc_83, %arg18, %34, %35) : (memref<288xf32>, index, f32, f32) -> memref<288xf32>
          scf.yield %39 : memref<288xf32>
        } else {
          scf.yield %arg20 : memref<288xf32>
        }
        scf.yield %36, %38 : memref<288xf32>, memref<288xf32>
      }
      %subview_60 = memref.subview %arg2[%arg16, %arg1, 0] [1, 1, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      %subview_61 = memref.subview %arg3[%arg16, %arg1, 0] [1, 1, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %reshape_59, %subview_61 : memref<288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %13#1, %subview_60 : memref<288xf32> to memref<288xf32, strided<[1], offset: ?>>
      %subview_62 = memref.subview %arg2[%arg16, 0, 0] [1, 256, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<256x288xf32, strided<[288, 1], offset: ?>>
      %subview_63 = memref.subview %arg3[%arg16, 0, 0] [1, 256, 288] [1, 1, 1] : memref<6x256x288xf32> to memref<256x288xf32, strided<[288, 1], offset: ?>>
      memref.copy %13#0, %alloc_19 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_62, %alloc_20 : memref<256x288xf32, strided<[288, 1], offset: ?>> to memref<256x288xf32>
      memref.copy %subview_63, %alloc_21 : memref<256x288xf32, strided<[288, 1], offset: ?>> to memref<256x288xf32>
      %14 = func.call @mha(%alloc_19, %alloc_20, %alloc_21, %arg1) : (memref<288xf32>, memref<256x288xf32>, memref<256x288xf32>, index) -> memref<288xf32>
      %subview_64 = memref.subview %arg9[%arg16, 0, 0] [1, 288, 288] [1, 1, 1] : memref<6x288x288xf32> to memref<288x288xf32, strided<[288, 1], offset: ?>>
      %reshape_65 = memref.reshape %14(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_22, %alloc_23 : memref<288x1xf32> to memref<288x1xf32>
      %15 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_23) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_64[%arg18, 0] [48, 288] [1, 1] : memref<288x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_65[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_24[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_24[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_25[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_25, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_66 = memref.reshape %15(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %16 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
      %17 = memref.get_global @__constant_2xi64_3 : memref<2xi64>
      %reshape_67 = memref.reshape %arg17(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      upmem.scatter %reshape_67[0, 48, #map] onto %16 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      %reshape_68 = memref.reshape %reshape_66(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      upmem.scatter %reshape_68[192, 48, #map] onto %16 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      %18 = memref.get_global @__constant_48x6xf32 : memref<48x6xf32>
      upmem.scatter %18[384, 48, #map] onto %16 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      upmem.launch_func  @dpu_kernels::@forward_3 %16 : !upmem.hierarchy<1x6x8> 
      upmem.gather %alloc_26[384, 48, #map] from %16 : memref<48x6xf32> from !upmem.hierarchy<1x6x8>
      %reshape_69 = memref.reshape %alloc_26(%8) : (memref<48x6xf32>, memref<1xi64>) -> memref<288xf32>
      upmem.free_dpus %16 : !upmem.hierarchy<1x6x8>
      %subview_70 = memref.subview %arg13[%arg16, 0] [1, 288] [1, 1] : memref<6x288xf32> to memref<288xf32, strided<[1], offset: ?>>
      memref.copy %reshape_69, %alloc_27 : memref<288xf32> to memref<288xf32>
      memref.copy %subview_70, %alloc_28 : memref<288xf32, strided<[1], offset: ?>> to memref<288xf32>
      %19 = func.call @rmsnorm(%alloc_27, %alloc_28) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
      %subview_71 = memref.subview %arg10[%arg16, 0, 0] [1, 768, 288] [1, 1, 1] : memref<6x768x288xf32> to memref<768x288xf32, strided<[288, 1], offset: ?>>
      %subview_72 = memref.subview %arg12[%arg16, 0, 0] [1, 768, 288] [1, 1, 1] : memref<6x768x288xf32> to memref<768x288xf32, strided<[288, 1], offset: ?>>
      %reshape_73 = memref.reshape %19(%6) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
      memref.copy %alloc_29, %alloc_30 : memref<768x1xf32> to memref<768x1xf32>
      %20 = scf.for %arg18 = %c0 to %c768 step %c48 iter_args(%arg19 = %alloc_30) -> (memref<768x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_71[%arg18, 0] [48, 288] [1, 1] : memref<768x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_73[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_31[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_31[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_32[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<768x1xf32>
        memref.copy %arg19, %alloc_83 : memref<768x1xf32> to memref<768x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<768x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_32, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<768x1xf32>
      }
      %21 = memref.get_global @__constant_1xi64_2 : memref<1xi64>
      %reshape_74 = memref.reshape %20(%21) : (memref<768x1xf32>, memref<1xi64>) -> memref<768xf32>
      memref.copy %alloc_33, %alloc_34 : memref<768x1xf32> to memref<768x1xf32>
      %22 = scf.for %arg18 = %c0 to %c768 step %c48 iter_args(%arg19 = %alloc_34) -> (memref<768x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_72[%arg18, 0] [48, 288] [1, 1] : memref<768x288xf32, strided<[288, 1], offset: ?>> to memref<48x288xf32, strided<[288, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c288 step %c1 {
          %29 = memref.load %reshape_73[%arg20, %c0] : memref<288x1xf32>
          memref.store %29, %alloc_35[%c0, %arg20] : memref<1x288xf32>
        }
        upmem.scatter %subview_82[0, 2304, #map] onto %28 : memref<48x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_35[9216, 2304, #map1] onto %28 : memref<1x288xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[18432, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_36[18432, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<768x1xf32>
        memref.copy %arg19, %alloc_83 : memref<768x1xf32> to memref<768x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<768x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_36, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<768x1xf32>
      }
      %reshape_75 = memref.reshape %22(%21) : (memref<768x1xf32>, memref<1xi64>) -> memref<768xf32>
      memref.copy %reshape_74, %alloc_37 : memref<768xf32> to memref<768xf32>
      %23 = scf.for %arg18 = %c0 to %c768 step %c1 iter_args(%arg19 = %alloc_37) -> (memref<768xf32>) {
        %27 = memref.load %arg19[%arg18] : memref<768xf32>
        %28 = memref.load %reshape_75[%arg18] : memref<768xf32>
        %29 = llvm.intr.exp(%27)  : (f32) -> f32
        %30 = arith.addf %29, %cst_0 : f32
        %31 = arith.divf %cst_0, %30 : f32
        %32 = arith.mulf %28, %31 : f32
        %alloc_82 = memref.alloc() {alignment = 64 : i64} : memref<768xf32>
        memref.copy %arg19, %alloc_82 : memref<768xf32> to memref<768xf32>
        memref.store %32, %alloc_82[%arg18] : memref<768xf32>
        scf.yield %alloc_82 : memref<768xf32>
      }
      %subview_76 = memref.subview %arg11[%arg16, 0, 0] [1, 288, 768] [1, 1, 1] : memref<6x288x768xf32> to memref<288x768xf32, strided<[768, 1], offset: ?>>
      %24 = memref.get_global @__constant_2xi64_4 : memref<2xi64>
      %reshape_77 = memref.reshape %23(%24) : (memref<768xf32>, memref<2xi64>) -> memref<768x1xf32>
      memref.copy %alloc_38, %alloc_39 : memref<288x1xf32> to memref<288x1xf32>
      %25 = scf.for %arg18 = %c0 to %c288 step %c48 iter_args(%arg19 = %alloc_39) -> (memref<288x1xf32>) {
        %27 = memref.get_global @__constant_48x1xf32 : memref<48x1xf32>
        %subview_82 = memref.subview %subview_76[%arg18, 0] [48, 768] [1, 1] : memref<288x768xf32, strided<[768, 1], offset: ?>> to memref<48x768xf32, strided<[768, 1], offset: ?>>
        %28 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
        scf.for %arg20 = %c0 to %c768 step %c1 {
          %29 = memref.load %reshape_77[%arg20, %c0] : memref<768x1xf32>
          memref.store %29, %alloc_40[%c0, %arg20] : memref<1x768xf32>
        }
        upmem.scatter %subview_82[0, 6144, #map] onto %28 : memref<48x768xf32, strided<[768, 1], offset: ?>> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %alloc_40[24576, 6144, #map1] onto %28 : memref<1x768xf32> onto !upmem.hierarchy<1x6x8>
        upmem.scatter %27[49152, 8, #map2] onto %28 : memref<48x1xf32> onto !upmem.hierarchy<1x6x8>
        upmem.launch_func  @dpu_kernels::@forward_6 %28 : !upmem.hierarchy<1x6x8> 
        upmem.gather %alloc_41[49152, 8, #map2] from %28 : memref<48x1xf32> from !upmem.hierarchy<1x6x8>
        upmem.free_dpus %28 : !upmem.hierarchy<1x6x8>
        %alloc_83 = memref.alloc() {alignment = 64 : i64} : memref<288x1xf32>
        memref.copy %arg19, %alloc_83 : memref<288x1xf32> to memref<288x1xf32>
        %subview_84 = memref.subview %alloc_83[%arg18, 0] [48, 1] [1, 1] : memref<288x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        memref.copy %alloc_41, %subview_84 : memref<48x1xf32> to memref<48x1xf32, strided<[1, 1], offset: ?>>
        scf.yield %alloc_83 : memref<288x1xf32>
      }
      %reshape_78 = memref.reshape %25(%8) : (memref<288x1xf32>, memref<1xi64>) -> memref<288xf32>
      %26 = upmem.alloc_dpus : !upmem.hierarchy<1x6x8>
      upmem.scatter %reshape_67[0, 48, #map] onto %26 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      %reshape_79 = memref.reshape %reshape_78(%17) : (memref<288xf32>, memref<2xi64>) -> memref<48x6xf32>
      upmem.scatter %reshape_79[192, 48, #map] onto %26 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      upmem.scatter %18[384, 48, #map] onto %26 : memref<48x6xf32> onto !upmem.hierarchy<1x6x8>
      upmem.launch_func  @dpu_kernels::@forward_3 %26 : !upmem.hierarchy<1x6x8> 
      %alloc_80 = memref.alloc() {alignment = 64 : i64} : memref<48x6xf32>
      upmem.gather %alloc_80[384, 48, #map] from %26 : memref<48x6xf32> from !upmem.hierarchy<1x6x8>
      %reshape_81 = memref.reshape %alloc_80(%8) : (memref<48x6xf32>, memref<1xi64>) -> memref<288xf32>
      upmem.free_dpus %26 : !upmem.hierarchy<1x6x8>
      scf.yield %reshape_81 : memref<288xf32>
    }
    %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    memref.copy %0, %alloc_42 : memref<288xf32> to memref<288xf32>
    %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    memref.copy %arg14, %alloc_43 : memref<288xf32> to memref<288xf32>
    %1 = call @rmsnorm(%alloc_42, %alloc_43) : (memref<288xf32>, memref<288xf32>) -> memref<288xf32>
    %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<32768x288xf32>
    scf.for %arg16 = %c0 to %c32768 step %c1 {
      scf.for %arg17 = %c0 to %c288 step %c1 {
        memref.store %cst, %alloc_44[%arg16, %arg17] : memref<32768x288xf32>
      }
    }
    %subview_45 = memref.subview %alloc_44[0, 0] [32000, 288] [1, 1] : memref<32768x288xf32> to memref<32000x288xf32, strided<[288, 1]>>
    memref.copy %arg15, %subview_45 : memref<32000x288xf32> to memref<32000x288xf32, strided<[288, 1]>>
    %2 = memref.get_global @__constant_2xi64_2 : memref<2xi64>
    %reshape = memref.reshape %1(%2) : (memref<288xf32>, memref<2xi64>) -> memref<288x1xf32>
    %alloc_46 = memref.alloc() {alignment = 64 : i64} : memref<32768x1xf32>
    %alloc_47 = memref.alloc() {alignment = 64 : i64} : memref<32768x1xf32>
    memref.copy %alloc_46, %alloc_47 : memref<32768x1xf32> to memref<32768x1xf32>
    %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<1x288xf32>
    %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<256x1xf32>
    %3 = scf.for %arg16 = %c0 to %c32768 step %c256 iter_args(%arg17 = %alloc_47) -> (memref<32768x1xf32>) {
      %5 = memref.get_global @__constant_256x1xf32 : memref<256x1xf32>
      %subview_52 = memref.subview %alloc_44[%arg16, 0] [256, 288] [1, 1] : memref<32768x288xf32> to memref<256x288xf32, strided<[288, 1], offset: ?>>
      %6 = upmem.alloc_dpus : !upmem.hierarchy<2x8x16>
      scf.for %arg18 = %c0 to %c288 step %c1 {
        %7 = memref.load %reshape[%arg18, %c0] : memref<288x1xf32>
        memref.store %7, %alloc_48[%c0, %arg18] : memref<1x288xf32>
      }
      upmem.scatter %subview_52[0, 4608, #map3] onto %6 : memref<256x288xf32, strided<[288, 1], offset: ?>> onto !upmem.hierarchy<2x8x16>
      upmem.scatter %alloc_48[18432, 4608, #map1] onto %6 : memref<1x288xf32> onto !upmem.hierarchy<2x8x16>
      upmem.scatter %5[36864, 16, #map4] onto %6 : memref<256x1xf32> onto !upmem.hierarchy<2x8x16>
      upmem.launch_func  @dpu_kernels::@forward_8 %6 : !upmem.hierarchy<2x8x16> 
      upmem.gather %alloc_49[36864, 16, #map4] from %6 : memref<256x1xf32> from !upmem.hierarchy<2x8x16>
      upmem.free_dpus %6 : !upmem.hierarchy<2x8x16>
      %alloc_53 = memref.alloc() {alignment = 64 : i64} : memref<32768x1xf32>
      memref.copy %arg17, %alloc_53 : memref<32768x1xf32> to memref<32768x1xf32>
      %subview_54 = memref.subview %alloc_53[%arg16, 0] [256, 1] [1, 1] : memref<32768x1xf32> to memref<256x1xf32, strided<[1, 1], offset: ?>>
      memref.copy %alloc_49, %subview_54 : memref<256x1xf32> to memref<256x1xf32, strided<[1, 1], offset: ?>>
      scf.yield %alloc_53 : memref<32768x1xf32>
    }
    %4 = memref.get_global @__constant_1xi64_3 : memref<1xi64>
    %reshape_50 = memref.reshape %3(%4) : (memref<32768x1xf32>, memref<1xi64>) -> memref<32768xf32>
    %subview_51 = memref.subview %reshape_50[0] [32000] [1] : memref<32768xf32> to memref<32000xf32, strided<[1]>>
    %cast = memref.cast %subview_51 : memref<32000xf32, strided<[1]>> to memref<32000xf32>
    return %cast : memref<32000xf32>
  }
  func.func @rot(%arg0: memref<288xf32>, %arg1: index, %arg2: f32, %arg3: f32) -> memref<288xf32> {
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg1, %c1 : index
    %1 = memref.load %arg0[%arg1] : memref<288xf32>
    %2 = memref.load %arg0[%0] : memref<288xf32>
    %3 = arith.mulf %1, %arg2 : f32
    %4 = arith.mulf %2, %arg3 : f32
    %5 = arith.subf %3, %4 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    memref.copy %arg0, %alloc : memref<288xf32> to memref<288xf32>
    memref.store %5, %alloc[%arg1] : memref<288xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    memref.copy %alloc, %alloc_0 : memref<288xf32> to memref<288xf32>
    memref.store %5, %alloc_0[%arg1] : memref<288xf32>
    return %alloc_0 : memref<288xf32>
  }
  func.func @mha(%arg0: memref<288xf32>, %arg1: memref<256x288xf32>, %arg2: memref<256x288xf32>, %arg3: index) -> memref<288xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c48 = arith.constant 48 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
    memref.copy %alloc, %alloc_0 : memref<288xf32> to memref<288xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<48xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x48xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x48xf32>
    %0 = scf.for %arg4 = %c0 to %c6 step %c1 iter_args(%arg5 = %alloc_0) -> (memref<288xf32>) {
      %1 = arith.muli %arg4, %c48 : index
      %subview = memref.subview %arg0[%1] [48] [1] : memref<288xf32> to memref<48xf32, strided<[1], offset: ?>>
      %subview_4 = memref.subview %arg1[0, %1] [256, 48] [1, 1] : memref<256x288xf32> to memref<256x48xf32, strided<[288, 1], offset: ?>>
      %subview_5 = memref.subview %arg2[0, %1] [256, 48] [1, 1] : memref<256x288xf32> to memref<256x48xf32, strided<[288, 1], offset: ?>>
      memref.copy %subview, %alloc_1 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
      memref.copy %subview_4, %alloc_2 : memref<256x48xf32, strided<[288, 1], offset: ?>> to memref<256x48xf32>
      memref.copy %subview_5, %alloc_3 : memref<256x48xf32, strided<[288, 1], offset: ?>> to memref<256x48xf32>
      %2 = func.call @attn(%alloc_1, %alloc_2, %alloc_3, %arg3) : (memref<48xf32>, memref<256x48xf32>, memref<256x48xf32>, index) -> memref<48xf32>
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<288xf32>
      memref.copy %arg5, %alloc_6 : memref<288xf32> to memref<288xf32>
      %subview_7 = memref.subview %alloc_6[%1] [48] [1] : memref<288xf32> to memref<48xf32, strided<[1], offset: ?>>
      memref.copy %2, %subview_7 : memref<48xf32> to memref<48xf32, strided<[1], offset: ?>>
      scf.yield %alloc_6 : memref<288xf32>
    }
    return %0 : memref<288xf32>
  }
  func.func @attn(%arg0: memref<48xf32>, %arg1: memref<256x48xf32>, %arg2: memref<256x48xf32>, %arg3: index) -> memref<48xf32> {
    %c256 = arith.constant 256 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 6.92820311 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    scf.for %arg4 = %c0 to %c256 step %c1 {
      memref.store %cst_0, %alloc[%arg4] : memref<256xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    memref.copy %alloc, %alloc_1 : memref<256xf32> to memref<256xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<48xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<8x6xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %0 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %alloc_1) -> (memref<256xf32>) {
      %subview = memref.subview %arg1[%arg4, 0] [1, 48] [1, 1] : memref<256x48xf32> to memref<48xf32, strided<[1], offset: ?>>
      %3 = upmem.alloc_dpus : !upmem.hierarchy<1x1x8>
      %4 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
      %reshape = memref.reshape %arg0(%4) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
      upmem.scatter %reshape[0, 48, #map5] onto %3 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      memref.copy %subview, %alloc_2 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
      %reshape_13 = memref.reshape %alloc_2(%4) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
      upmem.scatter %reshape_13[192, 48, #map5] onto %3 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      %5 = memref.get_global @__constant_8x6xf32 : memref<8x6xf32>
      upmem.scatter %5[384, 48, #map5] onto %3 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      upmem.launch_func  @dpu_kernels::@attn %3 : !upmem.hierarchy<1x1x8> 
      upmem.gather %alloc_3[384, 48, #map5] from %3 : memref<8x6xf32> from !upmem.hierarchy<1x1x8>
      %6 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
      %reshape_14 = memref.reshape %alloc_3(%6) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
      upmem.free_dpus %3 : !upmem.hierarchy<1x1x8>
      %7 = memref.get_global @__constant_xf32_0 : memref<f32>
      memref.copy %7, %alloc_5 : memref<f32> to memref<f32>
      scf.for %arg6 = %c0 to %c48 step %c1 {
        %14 = memref.load %reshape_14[%arg6] : memref<48xf32>
        %15 = memref.load %alloc_5[] : memref<f32>
        %16 = arith.addf %14, %15 : f32
        memref.store %16, %alloc_5[] : memref<f32>
      }
      %8 = memref.load %alloc_5[] : memref<f32>
      memref.store %8, %alloc_4[%c0] : memref<1xf32>
      memref.copy %7, %alloc_6 : memref<f32> to memref<f32>
      %9 = memref.load %alloc_4[%c0] : memref<1xf32>
      %10 = memref.load %alloc_6[] : memref<f32>
      %11 = arith.addf %9, %10 : f32
      memref.store %11, %alloc_6[] : memref<f32>
      %12 = memref.load %alloc_6[] : memref<f32>
      %13 = arith.divf %12, %cst : f32
      %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
      memref.copy %arg5, %alloc_15 : memref<256xf32> to memref<256xf32>
      memref.store %13, %alloc_15[%arg4] : memref<256xf32>
      scf.yield %alloc_15 : memref<256xf32>
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    memref.copy %0, %alloc_7 : memref<256xf32> to memref<256xf32>
    %1 = call @softmax(%alloc_7) : (memref<256xf32>) -> memref<256xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<48xf32>
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<48xf32>
    memref.copy %alloc_8, %alloc_9 : memref<48xf32> to memref<48xf32>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<48xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<8x6xf32>
    %2 = scf.for %arg4 = %c0 to %arg3 step %c1 iter_args(%arg5 = %alloc_9) -> (memref<48xf32>) {
      %subview = memref.subview %arg2[%arg4, 0] [1, 48] [1, 1] : memref<256x48xf32> to memref<48xf32, strided<[1], offset: ?>>
      %3 = memref.load %1[%arg4] : memref<256xf32>
      %4 = upmem.alloc_dpus : !upmem.hierarchy<1x1x8>
      %5 = memref.get_global @__constant_2xi64_0 : memref<2xi64>
      memref.copy %subview, %alloc_10 : memref<48xf32, strided<[1], offset: ?>> to memref<48xf32>
      %reshape = memref.reshape %alloc_10(%5) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
      upmem.scatter %reshape[0, 48, #map5] onto %4 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      memref.store %3, %alloc_11[%c0] : memref<8xf32>
      memref.store %3, %alloc_11[%c1] : memref<8xf32>
      memref.store %3, %alloc_11[%c2] : memref<8xf32>
      memref.store %3, %alloc_11[%c3] : memref<8xf32>
      memref.store %3, %alloc_11[%c4] : memref<8xf32>
      memref.store %3, %alloc_11[%c5] : memref<8xf32>
      memref.store %3, %alloc_11[%c6] : memref<8xf32>
      memref.store %3, %alloc_11[%c7] : memref<8xf32>
      upmem.scatter %alloc_11[192, 8, #map1] onto %4 : memref<8xf32> onto !upmem.hierarchy<1x1x8>
      %6 = memref.get_global @__constant_8x6xf32 : memref<8x6xf32>
      upmem.scatter %6[224, 48, #map5] onto %4 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      upmem.launch_func  @dpu_kernels::@attn_9 %4 : !upmem.hierarchy<1x1x8> 
      upmem.gather %alloc_12[224, 48, #map5] from %4 : memref<8x6xf32> from !upmem.hierarchy<1x1x8>
      %7 = memref.get_global @__constant_1xi64_0 : memref<1xi64>
      %reshape_13 = memref.reshape %alloc_12(%7) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
      upmem.free_dpus %4 : !upmem.hierarchy<1x1x8>
      %8 = upmem.alloc_dpus : !upmem.hierarchy<1x1x8>
      %reshape_14 = memref.reshape %reshape_13(%5) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
      upmem.scatter %reshape_14[0, 48, #map5] onto %8 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      %reshape_15 = memref.reshape %arg5(%5) : (memref<48xf32>, memref<2xi64>) -> memref<8x6xf32>
      upmem.scatter %reshape_15[192, 48, #map5] onto %8 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      upmem.scatter %6[384, 48, #map5] onto %8 : memref<8x6xf32> onto !upmem.hierarchy<1x1x8>
      upmem.launch_func  @dpu_kernels::@forward_3 %8 : !upmem.hierarchy<1x1x8> 
      %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<8x6xf32>
      upmem.gather %alloc_16[384, 48, #map5] from %8 : memref<8x6xf32> from !upmem.hierarchy<1x1x8>
      %reshape_17 = memref.reshape %alloc_16(%7) : (memref<8x6xf32>, memref<1xi64>) -> memref<48xf32>
      upmem.free_dpus %8 : !upmem.hierarchy<1x1x8>
      scf.yield %reshape_17 : memref<48xf32>
    }
    return %2 : memref<48xf32>
  }
  func.func @rmsnorm(%arg0: memref<288xf32>, %arg1: memref<288xf32>) -> memref<288xf32> {
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c288 = arith.constant 288 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 2.880000e+02 : f32
    %0 = upmem.alloc_dpus : !upmem.hierarchy<1x1x16>
    %1 = memref.get_global @__constant_2xi64_1 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    upmem.scatter %reshape[0, 288, #map5] onto %0 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.scatter %reshape[1152, 288, #map5] onto %0 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    %2 = memref.get_global @__constant_16x18xf32 : memref<16x18xf32>
    upmem.scatter %2[2304, 288, #map5] onto %0 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm %0 : !upmem.hierarchy<1x1x16> 
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x18xf32>
    upmem.gather %alloc[2304, 288, #map5] from %0 : memref<16x18xf32> from !upmem.hierarchy<1x1x16>
    %3 = memref.get_global @__constant_1xi64_1 : memref<1xi64>
    %reshape_1 = memref.reshape %alloc(%3) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    upmem.free_dpus %0 : !upmem.hierarchy<1x1x16>
    %4 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %4, %alloc_3 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c288 step %c1 {
      %17 = memref.load %reshape_1[%arg2] : memref<288xf32>
      %18 = memref.load %alloc_3[] : memref<f32>
      %19 = arith.addf %17, %18 : f32
      memref.store %19, %alloc_3[] : memref<f32>
    }
    %5 = memref.load %alloc_3[] : memref<f32>
    memref.store %5, %alloc_2[%c0] : memref<1xf32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %4, %alloc_4 : memref<f32> to memref<f32>
    %6 = memref.load %alloc_2[%c0] : memref<1xf32>
    %7 = memref.load %alloc_4[] : memref<f32>
    %8 = arith.addf %6, %7 : f32
    memref.store %8, %alloc_4[] : memref<f32>
    %9 = memref.load %alloc_4[] : memref<f32>
    %10 = arith.divf %9, %cst_0 : f32
    %11 = arith.addf %10, %cst : f32
    %12 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %13 = llvm.intr.sqrt(%11)  : (f32) -> f32
    %14 = llvm.fdiv %12, %13  : f32
    %15 = upmem.alloc_dpus : !upmem.hierarchy<1x1x16>
    upmem.scatter %reshape[0, 288, #map5] onto %15 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    memref.store %14, %alloc_5[%c0] : memref<16xf32>
    memref.store %14, %alloc_5[%c1] : memref<16xf32>
    memref.store %14, %alloc_5[%c2] : memref<16xf32>
    memref.store %14, %alloc_5[%c3] : memref<16xf32>
    memref.store %14, %alloc_5[%c4] : memref<16xf32>
    memref.store %14, %alloc_5[%c5] : memref<16xf32>
    memref.store %14, %alloc_5[%c6] : memref<16xf32>
    memref.store %14, %alloc_5[%c7] : memref<16xf32>
    memref.store %14, %alloc_5[%c8] : memref<16xf32>
    memref.store %14, %alloc_5[%c9] : memref<16xf32>
    memref.store %14, %alloc_5[%c10] : memref<16xf32>
    memref.store %14, %alloc_5[%c11] : memref<16xf32>
    memref.store %14, %alloc_5[%c12] : memref<16xf32>
    memref.store %14, %alloc_5[%c13] : memref<16xf32>
    memref.store %14, %alloc_5[%c14] : memref<16xf32>
    memref.store %14, %alloc_5[%c15] : memref<16xf32>
    upmem.scatter %alloc_5[1152, 16, #map1] onto %15 : memref<16xf32> onto !upmem.hierarchy<1x1x16>
    upmem.scatter %2[1216, 288, #map5] onto %15 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm_11 %15 : !upmem.hierarchy<1x1x16> 
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<16x18xf32>
    upmem.gather %alloc_6[1216, 288, #map5] from %15 : memref<16x18xf32> from !upmem.hierarchy<1x1x16>
    %reshape_7 = memref.reshape %alloc_6(%3) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    upmem.free_dpus %15 : !upmem.hierarchy<1x1x16>
    %16 = upmem.alloc_dpus : !upmem.hierarchy<1x1x16>
    %reshape_8 = memref.reshape %reshape_7(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    upmem.scatter %reshape_8[0, 288, #map5] onto %16 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    %reshape_9 = memref.reshape %arg1(%1) : (memref<288xf32>, memref<2xi64>) -> memref<16x18xf32>
    upmem.scatter %reshape_9[1152, 288, #map5] onto %16 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.scatter %2[2304, 288, #map5] onto %16 : memref<16x18xf32> onto !upmem.hierarchy<1x1x16>
    upmem.launch_func  @dpu_kernels::@rmsnorm %16 : !upmem.hierarchy<1x1x16> 
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<16x18xf32>
    upmem.gather %alloc_10[2304, 288, #map5] from %16 : memref<16x18xf32> from !upmem.hierarchy<1x1x16>
    %reshape_11 = memref.reshape %alloc_10(%3) : (memref<16x18xf32>, memref<1xi64>) -> memref<288xf32>
    upmem.free_dpus %16 : !upmem.hierarchy<1x1x16>
    return %reshape_11 : memref<288xf32>
  }
  func.func @softmax(%arg0: memref<256xf32>) -> memref<256xf32> {
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %17 = memref.load %arg0[%arg1] : memref<256xf32>
      %18 = memref.load %alloc_0[] : memref<f32>
      %19 = arith.maximumf %17, %18 : f32
      memref.store %19, %alloc_0[] : memref<f32>
    }
    %1 = memref.load %alloc_0[] : memref<f32>
    memref.store %1, %alloc[%c0] : memref<1xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    %2 = memref.load %alloc[%c0] : memref<1xf32>
    %3 = memref.load %alloc_1[] : memref<f32>
    %4 = arith.maximumf %2, %3 : f32
    memref.store %4, %alloc_1[] : memref<f32>
    %5 = memref.load %alloc_1[] : memref<f32>
    %6 = upmem.alloc_dpus : !upmem.hierarchy<1x8x16>
    %7 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%7) : (memref<256xf32>, memref<2xi64>) -> memref<128x2xf32>
    upmem.scatter %reshape[0, 32, #map3] onto %6 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    memref.store %5, %alloc_2[%c0] : memref<16xf32>
    memref.store %5, %alloc_2[%c1] : memref<16xf32>
    memref.store %5, %alloc_2[%c2] : memref<16xf32>
    memref.store %5, %alloc_2[%c3] : memref<16xf32>
    memref.store %5, %alloc_2[%c4] : memref<16xf32>
    memref.store %5, %alloc_2[%c5] : memref<16xf32>
    memref.store %5, %alloc_2[%c6] : memref<16xf32>
    memref.store %5, %alloc_2[%c7] : memref<16xf32>
    memref.store %5, %alloc_2[%c8] : memref<16xf32>
    memref.store %5, %alloc_2[%c9] : memref<16xf32>
    memref.store %5, %alloc_2[%c10] : memref<16xf32>
    memref.store %5, %alloc_2[%c11] : memref<16xf32>
    memref.store %5, %alloc_2[%c12] : memref<16xf32>
    memref.store %5, %alloc_2[%c13] : memref<16xf32>
    memref.store %5, %alloc_2[%c14] : memref<16xf32>
    memref.store %5, %alloc_2[%c15] : memref<16xf32>
    upmem.scatter %alloc_2[128, 16, #map1] onto %6 : memref<16xf32> onto !upmem.hierarchy<1x8x16>
    %8 = memref.get_global @__constant_128x2xf32 : memref<128x2xf32>
    upmem.scatter %8[192, 32, #map3] onto %6 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    upmem.launch_func  @dpu_kernels::@softmax %6 : !upmem.hierarchy<1x8x16> 
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<128x2xf32>
    upmem.gather %alloc_3[192, 32, #map3] from %6 : memref<128x2xf32> from !upmem.hierarchy<1x8x16>
    %9 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%9) : (memref<128x2xf32>, memref<1xi64>) -> memref<256xf32>
    upmem.free_dpus %6 : !upmem.hierarchy<1x8x16>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %17 = memref.load %reshape_4[%arg1] : memref<256xf32>
      %18 = llvm.intr.exp(%17)  : (f32) -> f32
      memref.store %18, %alloc_5[%arg1] : memref<256xf32>
    }
    %10 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %10, %alloc_7 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c256 step %c1 {
      %17 = memref.load %alloc_5[%arg1] : memref<256xf32>
      %18 = memref.load %alloc_7[] : memref<f32>
      %19 = arith.addf %17, %18 : f32
      memref.store %19, %alloc_7[] : memref<f32>
    }
    %11 = memref.load %alloc_7[] : memref<f32>
    memref.store %11, %alloc_6[%c0] : memref<1xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %10, %alloc_8 : memref<f32> to memref<f32>
    %12 = memref.load %alloc_6[%c0] : memref<1xf32>
    %13 = memref.load %alloc_8[] : memref<f32>
    %14 = arith.addf %12, %13 : f32
    memref.store %14, %alloc_8[] : memref<f32>
    %15 = memref.load %alloc_8[] : memref<f32>
    %16 = upmem.alloc_dpus : !upmem.hierarchy<1x8x16>
    %reshape_9 = memref.reshape %alloc_5(%7) : (memref<256xf32>, memref<2xi64>) -> memref<128x2xf32>
    upmem.scatter %reshape_9[0, 32, #map3] onto %16 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    memref.store %15, %alloc_10[%c0] : memref<16xf32>
    memref.store %15, %alloc_10[%c1] : memref<16xf32>
    memref.store %15, %alloc_10[%c2] : memref<16xf32>
    memref.store %15, %alloc_10[%c3] : memref<16xf32>
    memref.store %15, %alloc_10[%c4] : memref<16xf32>
    memref.store %15, %alloc_10[%c5] : memref<16xf32>
    memref.store %15, %alloc_10[%c6] : memref<16xf32>
    memref.store %15, %alloc_10[%c7] : memref<16xf32>
    memref.store %15, %alloc_10[%c8] : memref<16xf32>
    memref.store %15, %alloc_10[%c9] : memref<16xf32>
    memref.store %15, %alloc_10[%c10] : memref<16xf32>
    memref.store %15, %alloc_10[%c11] : memref<16xf32>
    memref.store %15, %alloc_10[%c12] : memref<16xf32>
    memref.store %15, %alloc_10[%c13] : memref<16xf32>
    memref.store %15, %alloc_10[%c14] : memref<16xf32>
    memref.store %15, %alloc_10[%c15] : memref<16xf32>
    upmem.scatter %alloc_10[128, 16, #map1] onto %16 : memref<16xf32> onto !upmem.hierarchy<1x8x16>
    upmem.scatter %8[192, 32, #map3] onto %16 : memref<128x2xf32> onto !upmem.hierarchy<1x8x16>
    upmem.launch_func  @dpu_kernels::@softmax_13 %16 : !upmem.hierarchy<1x8x16> 
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<128x2xf32>
    upmem.gather %alloc_11[192, 32, #map3] from %16 : memref<128x2xf32> from !upmem.hierarchy<1x8x16>
    %reshape_12 = memref.reshape %alloc_11(%9) : (memref<128x2xf32>, memref<1xi64>) -> memref<256xf32>
    upmem.free_dpus %16 : !upmem.hierarchy<1x8x16>
    return %reshape_12 : memref<256xf32>
  }
  upmem.module @dpu_kernels {
    upmem.func @forward() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1152 = arith.constant 1152 : index
      %2 = arith.muli %1, %c1152 : index
      %3 = arith.addi %0, %2 : index
      %c288 = arith.constant 288 : index
      %4 = upmem.pwram_alloc : memref<288xf32>
      %c9216 = arith.constant 9216 : index
      %5 = arith.addi %0, %c9216 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<288xf32>
      %8 = arith.addi %5, %c9216 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<f32>
      upmem.memcpy  mram_to_wram %4, %c288, %3 : memref<288xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c288, %6 : memref<288xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c288 step %c1 {
        %12 = memref.load %4[%arg0] : memref<288xf32>
        %13 = memref.load %7[%arg0] : memref<288xf32>
        %14 = memref.load %11[] : memref<f32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %15, %14 : f32
        memref.store %16, %11[] : memref<f32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<f32>, index, index
      upmem.return
    }
    upmem.func @forward_3() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c24 = arith.constant 24 : index
      %2 = arith.muli %1, %c24 : index
      %3 = arith.addi %0, %2 : index
      %c6 = arith.constant 6 : index
      %4 = upmem.pwram_alloc : memref<6xf32>
      %c192 = arith.constant 192 : index
      %5 = arith.addi %0, %c192 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<6xf32>
      %8 = arith.addi %5, %c192 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<6xf32>
      upmem.memcpy  mram_to_wram %4, %c6, %3 : memref<6xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c6, %6 : memref<6xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %11 = memref.load %4[%arg0] : memref<6xf32>
        %12 = memref.load %7[%arg0] : memref<6xf32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<6xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c6, %9 : memref<6xf32>, index, index
      upmem.return
    }
    upmem.func @forward_6() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c3072 = arith.constant 3072 : index
      %2 = arith.muli %1, %c3072 : index
      %3 = arith.addi %0, %2 : index
      %c768 = arith.constant 768 : index
      %4 = upmem.pwram_alloc : memref<768xf32>
      %c24576 = arith.constant 24576 : index
      %5 = arith.addi %0, %c24576 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<768xf32>
      %8 = arith.addi %5, %c24576 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<f32>
      upmem.memcpy  mram_to_wram %4, %c768, %3 : memref<768xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c768, %6 : memref<768xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c768 step %c1 {
        %12 = memref.load %4[%arg0] : memref<768xf32>
        %13 = memref.load %7[%arg0] : memref<768xf32>
        %14 = memref.load %11[] : memref<f32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %15, %14 : f32
        memref.store %16, %11[] : memref<f32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<f32>, index, index
      upmem.return
    }
    upmem.func @forward_8() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1152 = arith.constant 1152 : index
      %2 = arith.muli %1, %c1152 : index
      %3 = arith.addi %0, %2 : index
      %c288 = arith.constant 288 : index
      %4 = upmem.pwram_alloc : memref<288xf32>
      %c18432 = arith.constant 18432 : index
      %5 = arith.addi %0, %c18432 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<288xf32>
      %8 = arith.addi %5, %c18432 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<f32>
      upmem.memcpy  mram_to_wram %4, %c288, %3 : memref<288xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c288, %6 : memref<288xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c288 step %c1 {
        %12 = memref.load %4[%arg0] : memref<288xf32>
        %13 = memref.load %7[%arg0] : memref<288xf32>
        %14 = memref.load %11[] : memref<f32>
        %15 = arith.mulf %12, %13 : f32
        %16 = arith.addf %15, %14 : f32
        memref.store %16, %11[] : memref<f32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<f32>, index, index
      upmem.return
    }
    upmem.func @attn() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c24 = arith.constant 24 : index
      %2 = arith.muli %1, %c24 : index
      %3 = arith.addi %0, %2 : index
      %c6 = arith.constant 6 : index
      %4 = upmem.pwram_alloc : memref<6xf32>
      %c192 = arith.constant 192 : index
      %5 = arith.addi %0, %c192 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<6xf32>
      %8 = arith.addi %5, %c192 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<6xf32>
      upmem.memcpy  mram_to_wram %4, %c6, %3 : memref<6xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c6, %6 : memref<6xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %11 = memref.load %4[%arg0] : memref<6xf32>
        %12 = memref.load %7[%arg0] : memref<6xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<6xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c6, %9 : memref<6xf32>, index, index
      upmem.return
    }
    upmem.func @attn_9() attributes {num_tasklets = 8 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c24 = arith.constant 24 : index
      %2 = arith.muli %1, %c24 : index
      %3 = arith.addi %0, %2 : index
      %c6 = arith.constant 6 : index
      %4 = upmem.pwram_alloc : memref<6xf32>
      %c192 = arith.constant 192 : index
      %5 = arith.addi %0, %c192 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c32 = arith.constant 32 : index
      %7 = arith.addi %5, %c32 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<6xf32>
      upmem.memcpy  mram_to_wram %4, %c6, %3 : memref<6xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %10 = memref.load %4[%arg0] : memref<6xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<6xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c6, %8 : memref<6xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c72 = arith.constant 72 : index
      %2 = arith.muli %1, %c72 : index
      %3 = arith.addi %0, %2 : index
      %c18 = arith.constant 18 : index
      %4 = upmem.pwram_alloc : memref<18xf32>
      %c1152 = arith.constant 1152 : index
      %5 = arith.addi %0, %c1152 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<18xf32>
      %8 = arith.addi %5, %c1152 : index
      %9 = arith.addi %8, %2 : index
      %10 = upmem.pwram_alloc : memref<18xf32>
      upmem.memcpy  mram_to_wram %4, %c18, %3 : memref<18xf32>, index, index
      upmem.memcpy  mram_to_wram %7, %c18, %6 : memref<18xf32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c18 step %c1 {
        %11 = memref.load %4[%arg0] : memref<18xf32>
        %12 = memref.load %7[%arg0] : memref<18xf32>
        %13 = arith.mulf %11, %12 : f32
        memref.store %13, %10[%arg0] : memref<18xf32>
      }
      upmem.memcpy  wram_to_mram %10, %c18, %9 : memref<18xf32>, index, index
      upmem.return
    }
    upmem.func @rmsnorm_11() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c72 = arith.constant 72 : index
      %2 = arith.muli %1, %c72 : index
      %3 = arith.addi %0, %2 : index
      %c18 = arith.constant 18 : index
      %4 = upmem.pwram_alloc : memref<18xf32>
      %c1152 = arith.constant 1152 : index
      %5 = arith.addi %0, %c1152 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<18xf32>
      upmem.memcpy  mram_to_wram %4, %c18, %3 : memref<18xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c18 step %c1 {
        %10 = memref.load %4[%arg0] : memref<18xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.mulf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<18xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c18, %8 : memref<18xf32>, index, index
      upmem.return
    }
    upmem.func @softmax() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c8 = arith.constant 8 : index
      %2 = arith.muli %1, %c8 : index
      %3 = arith.addi %0, %2 : index
      %c2 = arith.constant 2 : index
      %4 = upmem.pwram_alloc : memref<2xf32>
      %c128 = arith.constant 128 : index
      %5 = arith.addi %0, %c128 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<2xf32>
      upmem.memcpy  mram_to_wram %4, %c2, %3 : memref<2xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %10 = memref.load %4[%arg0] : memref<2xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.subf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<2xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c2, %8 : memref<2xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_13() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c8 = arith.constant 8 : index
      %2 = arith.muli %1, %c8 : index
      %3 = arith.addi %0, %2 : index
      %c2 = arith.constant 2 : index
      %4 = upmem.pwram_alloc : memref<2xf32>
      %c128 = arith.constant 128 : index
      %5 = arith.addi %0, %c128 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<2xf32>
      upmem.memcpy  mram_to_wram %4, %c2, %3 : memref<2xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %10 = memref.load %4[%arg0] : memref<2xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.divf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<2xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c2, %8 : memref<2xf32>, index, index
      upmem.return
    }
  }
}

