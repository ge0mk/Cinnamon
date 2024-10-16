#map = affine_map<(d0, d1, d2) -> (d0 * 256 + d1 * 16 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (0)>
module {
  memref.global "private" constant @__constant_xf32_0 : memref<f32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1xi64 : memref<1xi64> = dense<131072> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1024x128xf32 : memref<1024x128xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[1024, 128]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_131072xf32 : memref<131072xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_xf32 : memref<f32> = dense<0xFF800000> {alignment = 64 : i64}
  func.func @softmax(%arg0: memref<131072xf32>) -> memref<131072xf32> {
    %c131072 = arith.constant 131072 : index
    %c128 = arith.constant 128 : index
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
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_xf32 : memref<f32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %9 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%9] [1024] [1] : memref<131072xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %11 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %12 = memref.load %alloc_0[] : memref<f32>
        %13 = arith.maximumf %11, %12 : f32
        memref.store %13, %alloc_0[] : memref<f32>
      }
      %10 = memref.load %alloc_0[] : memref<f32>
      memref.store %10, %alloc[%arg1] : memref<128xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %9 = memref.load %alloc[%arg1] : memref<128xf32>
      %10 = memref.load %alloc_1[] : memref<f32>
      %11 = arith.maximumf %9, %10 : f32
      memref.store %11, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = upmem.alloc_dpus : !upmem.hierarchy<4x16x16>
    %3 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<131072xf32>, memref<2xi64>) -> memref<1024x128xf32>
    upmem.scatter %reshape[0, 2048, #map] onto %2 : memref<1024x128xf32> onto !upmem.hierarchy<4x16x16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    memref.store %1, %alloc_2[%c0] : memref<16xf32>
    memref.store %1, %alloc_2[%c1] : memref<16xf32>
    memref.store %1, %alloc_2[%c2] : memref<16xf32>
    memref.store %1, %alloc_2[%c3] : memref<16xf32>
    memref.store %1, %alloc_2[%c4] : memref<16xf32>
    memref.store %1, %alloc_2[%c5] : memref<16xf32>
    memref.store %1, %alloc_2[%c6] : memref<16xf32>
    memref.store %1, %alloc_2[%c7] : memref<16xf32>
    memref.store %1, %alloc_2[%c8] : memref<16xf32>
    memref.store %1, %alloc_2[%c9] : memref<16xf32>
    memref.store %1, %alloc_2[%c10] : memref<16xf32>
    memref.store %1, %alloc_2[%c11] : memref<16xf32>
    memref.store %1, %alloc_2[%c12] : memref<16xf32>
    memref.store %1, %alloc_2[%c13] : memref<16xf32>
    memref.store %1, %alloc_2[%c14] : memref<16xf32>
    memref.store %1, %alloc_2[%c15] : memref<16xf32>
    upmem.scatter %alloc_2[8192, 16, #map1] onto %2 : memref<16xf32> onto !upmem.hierarchy<4x16x16>
    %4 = memref.get_global @__constant_1024x128xf32 : memref<1024x128xf32>
    upmem.scatter %4[8256, 2048, #map] onto %2 : memref<1024x128xf32> onto !upmem.hierarchy<4x16x16>
    upmem.launch_func  @dpu_kernels::@softmax %2 : !upmem.hierarchy<4x16x16> 
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1024x128xf32>
    upmem.gather %alloc_3[8256, 2048, #map] from %2 : memref<1024x128xf32> from !upmem.hierarchy<4x16x16>
    %5 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%5) : (memref<1024x128xf32>, memref<1xi64>) -> memref<131072xf32>
    upmem.free_dpus %2 : !upmem.hierarchy<4x16x16>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<131072xf32>
    scf.for %arg1 = %c0 to %c131072 step %c1 {
      %9 = memref.load %reshape_4[%arg1] : memref<131072xf32>
      %10 = llvm.intr.exp(%9)  : (f32) -> f32
      memref.store %10, %alloc_5[%arg1] : memref<131072xf32>
    }
    %6 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %9 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%9] [1024] [1] : memref<131072xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %6, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %11 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %12 = memref.load %alloc_7[] : memref<f32>
        %13 = arith.addf %11, %12 : f32
        memref.store %13, %alloc_7[] : memref<f32>
      }
      %10 = memref.load %alloc_7[] : memref<f32>
      memref.store %10, %alloc_6[%arg1] : memref<128xf32>
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %6, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %9 = memref.load %alloc_6[%arg1] : memref<128xf32>
      %10 = memref.load %alloc_8[] : memref<f32>
      %11 = arith.addf %9, %10 : f32
      memref.store %11, %alloc_8[] : memref<f32>
    }
    %7 = memref.load %alloc_8[] : memref<f32>
    %8 = upmem.alloc_dpus : !upmem.hierarchy<4x16x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<131072xf32>, memref<2xi64>) -> memref<1024x128xf32>
    upmem.scatter %reshape_9[0, 2048, #map] onto %8 : memref<1024x128xf32> onto !upmem.hierarchy<4x16x16>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    memref.store %7, %alloc_10[%c0] : memref<16xf32>
    memref.store %7, %alloc_10[%c1] : memref<16xf32>
    memref.store %7, %alloc_10[%c2] : memref<16xf32>
    memref.store %7, %alloc_10[%c3] : memref<16xf32>
    memref.store %7, %alloc_10[%c4] : memref<16xf32>
    memref.store %7, %alloc_10[%c5] : memref<16xf32>
    memref.store %7, %alloc_10[%c6] : memref<16xf32>
    memref.store %7, %alloc_10[%c7] : memref<16xf32>
    memref.store %7, %alloc_10[%c8] : memref<16xf32>
    memref.store %7, %alloc_10[%c9] : memref<16xf32>
    memref.store %7, %alloc_10[%c10] : memref<16xf32>
    memref.store %7, %alloc_10[%c11] : memref<16xf32>
    memref.store %7, %alloc_10[%c12] : memref<16xf32>
    memref.store %7, %alloc_10[%c13] : memref<16xf32>
    memref.store %7, %alloc_10[%c14] : memref<16xf32>
    memref.store %7, %alloc_10[%c15] : memref<16xf32>
    upmem.scatter %alloc_10[8192, 16, #map1] onto %8 : memref<16xf32> onto !upmem.hierarchy<4x16x16>
    upmem.scatter %4[8256, 2048, #map] onto %8 : memref<1024x128xf32> onto !upmem.hierarchy<4x16x16>
    upmem.launch_func  @dpu_kernels::@softmax_0 %8 : !upmem.hierarchy<4x16x16> 
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1024x128xf32>
    upmem.gather %alloc_11[8256, 2048, #map] from %8 : memref<1024x128xf32> from !upmem.hierarchy<4x16x16>
    %reshape_12 = memref.reshape %alloc_11(%5) : (memref<1024x128xf32>, memref<1xi64>) -> memref<131072xf32>
    upmem.free_dpus %8 : !upmem.hierarchy<4x16x16>
    return %reshape_12 : memref<131072xf32>
  }
  upmem.module @dpu_kernels {
    upmem.func @softmax() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c512 = arith.constant 512 : index
      %2 = arith.muli %1, %c512 : index
      %3 = arith.addi %0, %2 : index
      %c128 = arith.constant 128 : index
      %4 = upmem.pwram_alloc : memref<128xf32>
      %c8192 = arith.constant 8192 : index
      %5 = arith.addi %0, %c8192 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<128xf32>
      upmem.memcpy  mram_to_wram %4, %c128, %3 : memref<128xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c128 step %c1 {
        %10 = memref.load %4[%arg0] : memref<128xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.subf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<128xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c128, %8 : memref<128xf32>, index, index
      upmem.return
    }
    upmem.func @softmax_0() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c512 = arith.constant 512 : index
      %2 = arith.muli %1, %c512 : index
      %3 = arith.addi %0, %2 : index
      %c128 = arith.constant 128 : index
      %4 = upmem.pwram_alloc : memref<128xf32>
      %c8192 = arith.constant 8192 : index
      %5 = arith.addi %0, %c8192 : index
      %c2 = arith.constant 2 : index
      %6 = upmem.pwram_alloc : memref<f32>
      %c64 = arith.constant 64 : index
      %7 = arith.addi %5, %c64 : index
      %8 = arith.addi %7, %2 : index
      %9 = upmem.pwram_alloc : memref<128xf32>
      upmem.memcpy  mram_to_wram %4, %c128, %3 : memref<128xf32>, index, index
      upmem.memcpy  mram_to_wram %6, %c2, %5 : memref<f32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c128 step %c1 {
        %10 = memref.load %4[%arg0] : memref<128xf32>
        %11 = memref.load %6[] : memref<f32>
        %12 = arith.divf %10, %11 : f32
        memref.store %12, %9[%arg0] : memref<128xf32>
      }
      upmem.memcpy  wram_to_mram %9, %c128, %8 : memref<128xf32>, index, index
      upmem.return
    }
  }
}

