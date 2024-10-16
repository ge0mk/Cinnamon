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
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %arg0[%15] [1024] [1] : memref<131072xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %0, %alloc_0 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_0[] : memref<f32>
        %19 = arith.maximumf %17, %18 : f32
        memref.store %19, %alloc_0[] : memref<f32>
      }
      %16 = memref.load %alloc_0[] : memref<f32>
      memref.store %16, %alloc[%arg1] : memref<128xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %0, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %15 = memref.load %alloc[%arg1] : memref<128xf32>
      %16 = memref.load %alloc_1[] : memref<f32>
      %17 = arith.maximumf %15, %16 : f32
      memref.store %17, %alloc_1[] : memref<f32>
    }
    %1 = memref.load %alloc_1[] : memref<f32>
    %2 = cnm.workgroup : !cnm.workgroup<4x16x16>
    %3 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%3) : (memref<131072xf32>, memref<2xi64>) -> memref<1024x128xf32>
    %4 = cnm.alloc() for %2 : !cnm.buffer<128xf32 on 4x16x16, level 0>
    cnm.scatter %reshape into %4[#map] of %2 : memref<1024x128xf32> into !cnm.buffer<128xf32 on 4x16x16, level 0>
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
    %5 = cnm.alloc() for %2 : !cnm.buffer<f32 on 4x16x16, level 0>
    cnm.scatter %alloc_2 into %5[#map1] of %2 : memref<16xf32> into !cnm.buffer<f32 on 4x16x16, level 0>
    %6 = memref.get_global @__constant_1024x128xf32 : memref<1024x128xf32>
    %7 = cnm.alloc() for %2 : !cnm.buffer<128xf32 on 4x16x16, level 0>
    cnm.scatter %6 into %7[#map] of %2 : memref<1024x128xf32> into !cnm.buffer<128xf32 on 4x16x16, level 0>
    cnm.launch %2 in(%4, %5 : !cnm.buffer<128xf32 on 4x16x16, level 0>, !cnm.buffer<f32 on 4x16x16, level 0>) out(%7 : !cnm.buffer<128xf32 on 4x16x16, level 0>) on !cnm.workgroup<4x16x16> {
    ^bb0(%arg1: memref<128xf32>, %arg2: memref<f32>, %arg3: memref<128xf32>):
      %c0_13 = arith.constant 0 : index
      %c128_14 = arith.constant 128 : index
      %c1_15 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c128_14 step %c1_15 {
        %15 = memref.load %arg1[%arg4] : memref<128xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.subf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<128xf32>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1024x128xf32>
    cnm.gather %7[#map] of %2 into %alloc_3 : !cnm.buffer<128xf32 on 4x16x16, level 0> into memref<1024x128xf32>
    %8 = memref.get_global @__constant_1xi64 : memref<1xi64>
    %reshape_4 = memref.reshape %alloc_3(%8) : (memref<1024x128xf32>, memref<1xi64>) -> memref<131072xf32>
    cnm.free_workgroup %2 : !cnm.workgroup<4x16x16>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<131072xf32>
    scf.for %arg1 = %c0 to %c131072 step %c1 {
      %15 = memref.load %reshape_4[%arg1] : memref<131072xf32>
      %16 = math.exp %15 : f32
      memref.store %16, %alloc_5[%arg1] : memref<131072xf32>
    }
    %9 = memref.get_global @__constant_xf32_0 : memref<f32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %15 = arith.muli %arg1, %c1024 : index
      %subview = memref.subview %alloc_5[%15] [1024] [1] : memref<131072xf32> to memref<1024xf32, strided<[1], offset: ?>>
      memref.copy %9, %alloc_7 : memref<f32> to memref<f32>
      scf.for %arg2 = %c0 to %c1024 step %c1 {
        %17 = memref.load %subview[%arg2] : memref<1024xf32, strided<[1], offset: ?>>
        %18 = memref.load %alloc_7[] : memref<f32>
        %19 = arith.addf %17, %18 : f32
        memref.store %19, %alloc_7[] : memref<f32>
      }
      %16 = memref.load %alloc_7[] : memref<f32>
      memref.store %16, %alloc_6[%arg1] : memref<128xf32>
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %9, %alloc_8 : memref<f32> to memref<f32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %15 = memref.load %alloc_6[%arg1] : memref<128xf32>
      %16 = memref.load %alloc_8[] : memref<f32>
      %17 = arith.addf %15, %16 : f32
      memref.store %17, %alloc_8[] : memref<f32>
    }
    %10 = memref.load %alloc_8[] : memref<f32>
    %11 = cnm.workgroup : !cnm.workgroup<4x16x16>
    %reshape_9 = memref.reshape %alloc_5(%3) : (memref<131072xf32>, memref<2xi64>) -> memref<1024x128xf32>
    %12 = cnm.alloc() for %11 : !cnm.buffer<128xf32 on 4x16x16, level 0>
    cnm.scatter %reshape_9 into %12[#map] of %11 : memref<1024x128xf32> into !cnm.buffer<128xf32 on 4x16x16, level 0>
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<16xf32>
    memref.store %10, %alloc_10[%c0] : memref<16xf32>
    memref.store %10, %alloc_10[%c1] : memref<16xf32>
    memref.store %10, %alloc_10[%c2] : memref<16xf32>
    memref.store %10, %alloc_10[%c3] : memref<16xf32>
    memref.store %10, %alloc_10[%c4] : memref<16xf32>
    memref.store %10, %alloc_10[%c5] : memref<16xf32>
    memref.store %10, %alloc_10[%c6] : memref<16xf32>
    memref.store %10, %alloc_10[%c7] : memref<16xf32>
    memref.store %10, %alloc_10[%c8] : memref<16xf32>
    memref.store %10, %alloc_10[%c9] : memref<16xf32>
    memref.store %10, %alloc_10[%c10] : memref<16xf32>
    memref.store %10, %alloc_10[%c11] : memref<16xf32>
    memref.store %10, %alloc_10[%c12] : memref<16xf32>
    memref.store %10, %alloc_10[%c13] : memref<16xf32>
    memref.store %10, %alloc_10[%c14] : memref<16xf32>
    memref.store %10, %alloc_10[%c15] : memref<16xf32>
    %13 = cnm.alloc() for %11 : !cnm.buffer<f32 on 4x16x16, level 0>
    cnm.scatter %alloc_10 into %13[#map1] of %11 : memref<16xf32> into !cnm.buffer<f32 on 4x16x16, level 0>
    %14 = cnm.alloc() for %11 : !cnm.buffer<128xf32 on 4x16x16, level 0>
    cnm.scatter %6 into %14[#map] of %11 : memref<1024x128xf32> into !cnm.buffer<128xf32 on 4x16x16, level 0>
    cnm.launch %11 in(%12, %13 : !cnm.buffer<128xf32 on 4x16x16, level 0>, !cnm.buffer<f32 on 4x16x16, level 0>) out(%14 : !cnm.buffer<128xf32 on 4x16x16, level 0>) on !cnm.workgroup<4x16x16> {
    ^bb0(%arg1: memref<128xf32>, %arg2: memref<f32>, %arg3: memref<128xf32>):
      %c0_13 = arith.constant 0 : index
      %c128_14 = arith.constant 128 : index
      %c1_15 = arith.constant 1 : index
      scf.for %arg4 = %c0_13 to %c128_14 step %c1_15 {
        %15 = memref.load %arg1[%arg4] : memref<128xf32>
        %16 = memref.load %arg2[] : memref<f32>
        %17 = arith.divf %15, %16 : f32
        memref.store %17, %arg3[%arg4] : memref<128xf32>
      }
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1024x128xf32>
    cnm.gather %14[#map] of %11 into %alloc_11 : !cnm.buffer<128xf32 on 4x16x16, level 0> into memref<1024x128xf32>
    %reshape_12 = memref.reshape %alloc_11(%8) : (memref<1024x128xf32>, memref<1xi64>) -> memref<131072xf32>
    cnm.free_workgroup %11 : !cnm.workgroup<4x16x16>
    return %reshape_12 : memref<131072xf32>
  }
}

