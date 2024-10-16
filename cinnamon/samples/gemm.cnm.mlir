#map = affine_map<(d0, d1, d2) -> (0)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 16 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (0, d1 * 16 + d2)>
module {
  memref.global "private" constant @__constant_1x512xi32 : memref<1x512xi32> = dense<0> {alignment = 64 : i64}
  func.func @main() {
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
    memref.copy %alloc_1, %alloc_2 : memref<1024x1024xi32> to memref<1024x1024xi32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x512xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<512x256xi32>
    %0 = scf.for %arg0 = %c0 to %c1024 step %c1 iter_args(%arg1 = %alloc_2) -> (memref<1024x1024xi32>) {
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
      memref.copy %arg1, %alloc_5 : memref<1024x1024xi32> to memref<1024x1024xi32>
      %1 = scf.for %arg2 = %c0 to %c1024 step %c512 iter_args(%arg3 = %alloc_5) -> (memref<1024x1024xi32>) {
        %2 = memref.get_global @__constant_1x512xi32 : memref<1x512xi32>
        memref.copy %2, %alloc_3 : memref<1x512xi32> to memref<1x512xi32>
        %3 = scf.for %arg4 = %c0 to %c1024 step %c256 iter_args(%arg5 = %alloc_3) -> (memref<1x512xi32>) {
          %subview_7 = memref.subview %alloc[%arg0, %arg4] [1, 256] [1, 1] : memref<1024x1024xi32> to memref<1x256xi32, strided<[1024, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_0[%arg4, %arg2] [256, 512] [1, 1] : memref<1024x1024xi32> to memref<256x512xi32, strided<[1024, 1], offset: ?>>
          %4 = cnm.workgroup : !cnm.workgroup<2x16x16>
          scf.for %arg6 = %c0 to %c512 step %c1 {
            scf.for %arg7 = %c0 to %c256 step %c1 {
              %8 = memref.load %subview_8[%arg7, %arg6] : memref<256x512xi32, strided<[1024, 1], offset: ?>>
              memref.store %8, %alloc_4[%arg6, %arg7] : memref<512x256xi32>
            }
          }
          %5 = cnm.alloc() for %4 : !cnm.buffer<256xi32 on 2x16x16, level 0>
          %6 = cnm.alloc() for %4 : !cnm.buffer<256xi32 on 2x16x16, level 0>
          %7 = cnm.alloc() for %4 : !cnm.buffer<i32 on 2x16x16, level 0>
          cnm.scatter %subview_7 into %5[#map] of %4 : memref<1x256xi32, strided<[1024, 1], offset: ?>> into !cnm.buffer<256xi32 on 2x16x16, level 0>
          cnm.scatter %alloc_4 into %6[#map1] of %4 : memref<512x256xi32> into !cnm.buffer<256xi32 on 2x16x16, level 0>
          cnm.scatter %arg5 into %7[#map2] of %4 : memref<1x512xi32> into !cnm.buffer<i32 on 2x16x16, level 0>
          cnm.launch %4 in(%5, %6 : !cnm.buffer<256xi32 on 2x16x16, level 0>, !cnm.buffer<256xi32 on 2x16x16, level 0>) out(%7 : !cnm.buffer<i32 on 2x16x16, level 0>) on !cnm.workgroup<2x16x16> {
          ^bb0(%arg6: memref<256xi32>, %arg7: memref<256xi32>, %arg8: memref<i32>):
            %c0_10 = arith.constant 0 : index
            %c256_11 = arith.constant 256 : index
            %c1_12 = arith.constant 1 : index
            scf.for %arg9 = %c0_10 to %c256_11 step %c1_12 {
              %8 = memref.load %arg6[%arg9] : memref<256xi32>
              %9 = memref.load %arg7[%arg9] : memref<256xi32>
              %10 = memref.load %arg8[] : memref<i32>
              %11 = arith.muli %8, %9 : i32
              %12 = arith.addi %11, %10 : i32
              memref.store %12, %arg8[] : memref<i32>
            }
          }
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x512xi32>
          cnm.gather %7[#map2] of %4 into %alloc_9 : !cnm.buffer<i32 on 2x16x16, level 0> into memref<1x512xi32>
          cnm.free_workgroup %4 : !cnm.workgroup<2x16x16>
          scf.yield %alloc_9 : memref<1x512xi32>
        }
        %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xi32>
        memref.copy %arg3, %alloc_6 : memref<1024x1024xi32> to memref<1024x1024xi32>
        %subview = memref.subview %alloc_6[%arg0, %arg2] [1, 512] [1, 1] : memref<1024x1024xi32> to memref<1x512xi32, strided<[1024, 1], offset: ?>>
        memref.copy %3, %subview : memref<1x512xi32> to memref<1x512xi32, strided<[1024, 1], offset: ?>>
        scf.yield %alloc_6 : memref<1024x1024xi32>
      }
      scf.yield %1 : memref<1024x1024xi32>
    }
    return
  }
}

