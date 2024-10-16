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
          %4 = upmem.alloc_dpus : !upmem.hierarchy<2x16x16>
          scf.for %arg6 = %c0 to %c512 step %c1 {
            scf.for %arg7 = %c0 to %c256 step %c1 {
              %5 = memref.load %subview_8[%arg7, %arg6] : memref<256x512xi32, strided<[1024, 1], offset: ?>>
              memref.store %5, %alloc_4[%arg6, %arg7] : memref<512x256xi32>
            }
          }
          upmem.scatter %subview_7[0, 4096, #map] onto %4 : memref<1x256xi32, strided<[1024, 1], offset: ?>> onto !upmem.hierarchy<2x16x16>
          upmem.scatter %alloc_4[16384, 4096, #map1] onto %4 : memref<512x256xi32> onto !upmem.hierarchy<2x16x16>
          upmem.scatter %arg5[32768, 16, #map2] onto %4 : memref<1x512xi32> onto !upmem.hierarchy<2x16x16>
          upmem.launch_func  @dpu_kernels::@main %4 : !upmem.hierarchy<2x16x16> 
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x512xi32>
          upmem.gather %alloc_9[32768, 16, #map2] from %4 : memref<1x512xi32> from !upmem.hierarchy<2x16x16>
          upmem.free_dpus %4 : !upmem.hierarchy<2x16x16>
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
  upmem.module @dpu_kernels {
    upmem.func @main() attributes {num_tasklets = 16 : i64} {
      %0 = upmem.dpu_heap_base_addr : index
      %1 = upmem.tasklet_id : index
      %c1024 = arith.constant 1024 : index
      %2 = arith.muli %1, %c1024 : index
      %3 = arith.addi %0, %2 : index
      %c256 = arith.constant 256 : index
      %4 = upmem.pwram_alloc : memref<256xi32>
      %c16384 = arith.constant 16384 : index
      %5 = arith.addi %0, %c16384 : index
      %6 = arith.addi %5, %2 : index
      %7 = upmem.pwram_alloc : memref<256xi32>
      %8 = arith.addi %5, %c16384 : index
      %c8 = arith.constant 8 : index
      %9 = arith.muli %1, %c8 : index
      %10 = arith.addi %8, %9 : index
      %c2 = arith.constant 2 : index
      %11 = upmem.pwram_alloc : memref<i32>
      upmem.memcpy  mram_to_wram %4, %c256, %3 : memref<256xi32>, index, index
      upmem.memcpy  mram_to_wram %7, %c256, %6 : memref<256xi32>, index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c256 step %c1 {
        %12 = memref.load %4[%arg0] : memref<256xi32>
        %13 = memref.load %7[%arg0] : memref<256xi32>
        %14 = memref.load %11[] : memref<i32>
        %15 = arith.muli %12, %13 : i32
        %16 = arith.addi %15, %14 : i32
        memref.store %16, %11[] : memref<i32>
      }
      upmem.memcpy  wram_to_mram %11, %c2, %10 : memref<i32>, index, index
      upmem.return
    }
  }
}

