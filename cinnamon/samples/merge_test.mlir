#mapt = affine_map<(d0, d1, d2) -> (d1 * 16 + d2)>
module {
  memref.global "private" constant @__tconstant_1xi64 : memref<1xi64> = dense<1024> {alignment = 64 : i64}
  memref.global "private" constant @__tconstant_256x4xi32 : memref<256x4xi32> = dense<0> {alignment = 64 : i64}
  memref.global "private" constant @__tconstant_2xi64 : memref<2xi64> = dense<[256, 4]> {alignment = 64 : i64}
  memref.global "private" constant @__tconstant_1024xi32 : memref<1024xi32> = dense<0> {alignment = 64 : i64}
  func.func @test_0(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) {
    %0 = cnm.workgroup : !cnm.workgroup<1x16x1__tconstant_2xi646>
    %6 = cnm.workgroup : !cnm.workgroup<1x16x16>
    %10 = cnm.workgroup : !cnm.workgroup<1x16x16>

    %1 = memref.get_global @__tconstant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>

    %2 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %2[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %3 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_0 into %3[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %4 = memref.get_global @__tconstant_256x4xi32 : memref<256x4xi32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %5[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>

    %7 = cnm.alloc() for %6 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %7[#mapt] of %6 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_1 = memref.reshape %arg2(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %8 = cnm.alloc() for %6 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_1 into %8[#mapt] of %6 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %9 = cnm.alloc() for %6 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %9[#mapt] of %6 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>

    %11 = cnm.alloc() for %10 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %11[#mapt] of %10 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_3 = memref.reshape %arg3(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %12 = cnm.alloc() for %10 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_3 into %12[#mapt] of %10 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %13 = cnm.alloc() for %10 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %13[#mapt] of %10 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>

    cnm.launch %0 in(%2, %3 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%5 : !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg7 = %c0 to %c4 step %c1 {
        %14 = memref.load %arg4[%arg7] : memref<4xi32>
        %15 = memref.load %arg5[%arg7] : memref<4xi32>
        %16 = arith.addi %14, %15 : i32
        memref.store %16, %arg6[%arg7] : memref<4xi32>
      }
    }

    cnm.launch %6 in(%7, %8 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%9 : !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg7 = %c0 to %c4 step %c1 {
        %14 = memref.load %arg4[%arg7] : memref<4xi32>
        %15 = memref.load %arg5[%arg7] : memref<4xi32>
        %16 = arith.addi %14, %15 : i32
        memref.store %16, %arg6[%arg7] : memref<4xi32>
      }
    }

    cnm.launch %10 in(%11, %12 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%13 : !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg7 = %c0 to %c4 step %c1 {
        %14 = memref.load %arg4[%arg7] : memref<4xi32>
        %15 = memref.load %arg5[%arg7] : memref<4xi32>
        %16 = arith.addi %14, %15 : i32
        memref.store %16, %arg6[%arg7] : memref<4xi32>
      }
    }

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %5[#mapt] of %0 into %alloc : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %9[#mapt] of %6 into %alloc_2 : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %13[#mapt] of %10 into %alloc_4 : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>

    cnm.free_workgroup %0 : !cnm.workgroup<1x16x16>
    cnm.free_workgroup %6 : !cnm.workgroup<1x16x16>
    cnm.free_workgroup %10 : !cnm.workgroup<1x16x16>

    return
  }

  func.func @test_1(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) {
    %0 = cnm.workgroup : !cnm.workgroup<1x16x16>
    %1 = memref.get_global @__tconstant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %2[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %3 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_0 into %3[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %4 = memref.get_global @__tconstant_256x4xi32 : memref<256x4xi32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %5[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %6 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %6[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_1 = memref.reshape %arg2(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %7 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_1 into %7[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %8 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %8[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %9 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %9[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_2 = memref.reshape %arg3(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %10 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_2 into %10[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %11 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %11[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.launch %0 in(%2, %3 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%5 : !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg7 = %c0 to %c4 step %c1 {
        %12 = memref.load %arg4[%arg7] : memref<4xi32>
        %13 = memref.load %arg5[%arg7] : memref<4xi32>
        %14 = arith.addi %12, %13 : i32
        memref.store %14, %arg6[%arg7] : memref<4xi32>
      }
    }
    cnm.launch %0 in(%6, %7 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%8 : !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg7 = %c0 to %c4 step %c1 {
        %12 = memref.load %arg4[%arg7] : memref<4xi32>
        %13 = memref.load %arg5[%arg7] : memref<4xi32>
        %14 = arith.addi %12, %13 : i32
        memref.store %14, %arg6[%arg7] : memref<4xi32>
      }
    }
    cnm.launch %0 in(%9, %10 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%11 : !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg7 = %c0 to %c4 step %c1 {
        %12 = memref.load %arg4[%arg7] : memref<4xi32>
        %13 = memref.load %arg5[%arg7] : memref<4xi32>
        %14 = arith.addi %12, %13 : i32
        memref.store %14, %arg6[%arg7] : memref<4xi32>
      }
    }
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %5[#mapt] of %0 into %alloc : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %8[#mapt] of %0 into %alloc_3 : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %11[#mapt] of %0 into %alloc_4 : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    cnm.free_workgroup %0 : !cnm.workgroup<1x16x16>
    return
  }

  func.func @test_2(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) {
    %0 = cnm.workgroup : !cnm.workgroup<1x16x16>
    %1 = memref.get_global @__tconstant_2xi64 : memref<2xi64>
    %reshape = memref.reshape %arg0(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %2 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %2[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_0 = memref.reshape %arg1(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %3 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_0 into %3[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %4 = memref.get_global @__tconstant_256x4xi32 : memref<256x4xi32>
    %5 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %5[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %6 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %6[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_1 = memref.reshape %arg2(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %7 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_1 into %7[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %8 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %8[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %9 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape into %9[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %reshape_2 = memref.reshape %arg3(%1) : (memref<1024xi32>, memref<2xi64>) -> memref<256x4xi32>
    %10 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %reshape_2 into %10[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    %11 = cnm.alloc() for %0 : !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.scatter %4 into %11[#mapt] of %0 : memref<256x4xi32> into !cnm.buffer<4xi32 on 1x16x16, level 0>
    cnm.launch %0 in(%2, %3, %6, %7, %9, %10 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) out(%5, %8, %11 : !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>, !cnm.buffer<4xi32 on 1x16x16, level 0>) on !cnm.workgroup<1x16x16> {
    ^bb0(%arg4: memref<4xi32>, %arg5: memref<4xi32>, %arg6: memref<4xi32>, %arg7: memref<4xi32>, %arg8: memref<4xi32>, %arg9: memref<4xi32>, %arg10: memref<4xi32>, %arg11: memref<4xi32>, %arg12: memref<4xi32>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg13 = %c0 to %c4 step %c1 {
        %12 = memref.load %arg4[%arg13] : memref<4xi32>
        %13 = memref.load %arg5[%arg13] : memref<4xi32>
        %14 = arith.addi %12, %13 : i32
        memref.store %14, %arg10[%arg13] : memref<4xi32>
      }
      %c0_5 = arith.constant 0 : index
      %c4_6 = arith.constant 4 : index
      %c1_7 = arith.constant 1 : index
      scf.for %arg13 = %c0_5 to %c4_6 step %c1_7 {
        %12 = memref.load %arg6[%arg13] : memref<4xi32>
        %13 = memref.load %arg7[%arg13] : memref<4xi32>
        %14 = arith.addi %12, %13 : i32
        memref.store %14, %arg11[%arg13] : memref<4xi32>
      }
      %c0_8 = arith.constant 0 : index
      %c4_9 = arith.constant 4 : index
      %c1_10 = arith.constant 1 : index
      scf.for %arg13 = %c0_8 to %c4_9 step %c1_10 {
        %12 = memref.load %arg8[%arg13] : memref<4xi32>
        %13 = memref.load %arg9[%arg13] : memref<4xi32>
        %14 = arith.addi %12, %13 : i32
        memref.store %14, %arg12[%arg13] : memref<4xi32>
      }
    }
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %5[#mapt] of %0 into %alloc : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %8[#mapt] of %0 into %alloc_3 : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x4xi32>
    cnm.gather %11[#mapt] of %0 into %alloc_4 : !cnm.buffer<4xi32 on 1x16x16, level 0> into memref<256x4xi32>
    cnm.free_workgroup %0 : !cnm.workgroup<1x16x16>
    return
  }
}
