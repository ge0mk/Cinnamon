module {
  func.func @mm_dimm1_nopt(%arg0: tensor<1x1024xi32>, %arg1: tensor<1024x512xi32>) {
    cinm.compute attributes {workgroupShape = array<i64: 1, 128>, maxDpuBufferSize=128} {

      %0 = tensor.empty() : tensor<1x512xi32>
      %1 = affine.for %arg2 = 0 to 512 step 128 iter_args(%arg3 = %0) -> (tensor<1x512xi32>) {
        %c0_i32 = arith.constant 0 : i32
        %splat = tensor.splat %c0_i32 : tensor<1x128xi32>
        %2 = affine.for %arg4 = 0 to 1024 step 32 iter_args(%arg5 = %splat) -> (tensor<1x128xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg4] [1, 32] [1, 1] : tensor<1x1024xi32> to tensor<1x32xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [32, 128] [1, 1] : tensor<1024x512xi32> to tensor<32x128xi32>
          %3 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {cinm.notile} : (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
          affine.yield %3 : tensor<1x128xi32>
        }
        %inserted_slice = tensor.insert_slice %2 into %arg3[0, %arg2] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<1x512xi32>
        affine.yield %inserted_slice : tensor<1x512xi32>
      }
      cinm.yield
    }
    return
  }
  func.func @mm_dimm1_opt(%arg0: tensor<16x64xi32>, %arg1: tensor<64x512xi32>) {
    cinm.compute attributes {workgroupShape = array<i64: 1, 128>} {
      %0 = tensor.empty() : tensor<16x512xi32>
      %1 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %0) -> (tensor<16x512xi32>) {
        %2 = affine.for %arg4 = 0 to 512 step 128 iter_args(%arg5 = %arg3) -> (tensor<16x512xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 64] [1, 1] : tensor<16x64xi32> to tensor<1x64xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [64, 128] [1, 1] : tensor<64x512xi32> to tensor<64x128xi32>
          %3 = tensor.empty() : tensor<1x128xi32>
          %4 = affine.for %arg6 = 0 to 128 step 128 iter_args(%arg7 = %3) -> (tensor<1x128xi32>) {
            %c0_i32 = arith.constant 0 : i32
            %splat = tensor.splat %c0_i32 : tensor<1x128xi32>
            %5 = affine.for %arg8 = 0 to 64 step 32 iter_args(%arg9 = %splat) -> (tensor<1x128xi32>) {
              %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg8] [1, 32] [1, 1] : tensor<1x64xi32> to tensor<1x32xi32>
              %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg6, %arg8] [32, 128] [1, 1] : tensor<64x128xi32> to tensor<32x128xi32>
              %6 = cinm.op.gemm %extracted_slice_2, %extracted_slice_3 plus %arg9 {notile = #cinm.notile} : (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
              affine.yield %6 : tensor<1x128xi32>
            }
            %inserted_slice_1 = tensor.insert_slice %5 into %arg7[0, %arg6] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<1x128xi32>
            affine.yield %inserted_slice_1 : tensor<1x128xi32>
          }
          %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<16x512xi32>
          affine.yield %inserted_slice : tensor<16x512xi32>
        }
        affine.yield %2 : tensor<16x512xi32>
      }
      cinm.yield
    }
    return
  }
  func.func @mm_dimm2_nopt(%arg0: tensor<1x1024xi32>, %arg1: tensor<1024x256xi32>) {
    cinm.compute attributes {workgroupShape = array<i64: 2, 128>} {
      %0 = tensor.empty() : tensor<1x256xi32>
      %1 = affine.for %arg2 = 0 to 256 step 256 iter_args(%arg3 = %0) -> (tensor<1x256xi32>) {
        %c0_i32 = arith.constant 0 : i32
        %splat = tensor.splat %c0_i32 : tensor<1x256xi32>
        %2 = affine.for %arg4 = 0 to 1024 step 32 iter_args(%arg5 = %splat) -> (tensor<1x256xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg4] [1, 32] [1, 1] : tensor<1x1024xi32> to tensor<1x32xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [32, 256] [1, 1] : tensor<1024x256xi32> to tensor<32x256xi32>
          %3 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {notile = #cinm.notile} : (tensor<1x32xi32>, tensor<32x256xi32>) -> tensor<1x256xi32>
          affine.yield %3 : tensor<1x256xi32>
        }
        %inserted_slice = tensor.insert_slice %2 into %arg3[0, %arg2] [1, 256] [1, 1] : tensor<1x256xi32> into tensor<1x256xi32>
        affine.yield %inserted_slice : tensor<1x256xi32>
      }
      cinm.yield
    }
    return
  }
  func.func @mm_dimm2_opt(%arg0: tensor<16x64xi32>, %arg1: tensor<64x256xi32>) {
    cinm.compute attributes {workgroupShape = array<i64: 2, 128>} {
      %0 = tensor.empty() : tensor<16x256xi32>
      %1 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %0) -> (tensor<16x256xi32>) {
        %2 = affine.for %arg4 = 0 to 256 step 256 iter_args(%arg5 = %arg3) -> (tensor<16x256xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 64] [1, 1] : tensor<16x64xi32> to tensor<1x64xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [64, 256] [1, 1] : tensor<64x256xi32> to tensor<64x256xi32>
          %3 = tensor.empty() : tensor<1x256xi32>
          %4 = affine.for %arg6 = 0 to 256 step 256 iter_args(%arg7 = %3) -> (tensor<1x256xi32>) {
            %c0_i32 = arith.constant 0 : i32
            %splat = tensor.splat %c0_i32 : tensor<1x256xi32>
            %5 = affine.for %arg8 = 0 to 64 step 32 iter_args(%arg9 = %splat) -> (tensor<1x256xi32>) {
              %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg8] [1, 32] [1, 1] : tensor<1x64xi32> to tensor<1x32xi32>
              %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg6, %arg8] [32, 256] [1, 1] : tensor<64x256xi32> to tensor<32x256xi32>
              %6 = cinm.op.gemm %extracted_slice_2, %extracted_slice_3 plus %arg9 {notile = #cinm.notile} : (tensor<1x32xi32>, tensor<32x256xi32>) -> tensor<1x256xi32>
              affine.yield %6 : tensor<1x256xi32>
            }
            %inserted_slice_1 = tensor.insert_slice %5 into %arg7[0, %arg6] [1, 256] [1, 1] : tensor<1x256xi32> into tensor<1x256xi32>
            affine.yield %inserted_slice_1 : tensor<1x256xi32>
          }
          %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 256] [1, 1] : tensor<1x256xi32> into tensor<16x256xi32>
          affine.yield %inserted_slice : tensor<16x256xi32>
        }
        affine.yield %2 : tensor<16x256xi32>
      }
      cinm.yield
    }
    return
  }
  func.func @mm_dimm4_nopt(%arg0: tensor<1x1024xi32>, %arg1: tensor<1024x128xi32>) {
    cinm.compute attributes {workgroupShape = array<i64: 4, 128>} {
      %0 = tensor.empty() : tensor<1x128xi32>
      %1 = affine.for %arg2 = 0 to 128 step 128 iter_args(%arg3 = %0) -> (tensor<1x128xi32>) {
        %c0_i32 = arith.constant 0 : i32
        %splat = tensor.splat %c0_i32 : tensor<1x128xi32>
        %2 = affine.for %arg4 = 0 to 1024 step 32 iter_args(%arg5 = %splat) -> (tensor<1x128xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[0, %arg4] [1, 32] [1, 1] : tensor<1x1024xi32> to tensor<1x32xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, %arg4] [32, 128] [1, 1] : tensor<1024x128xi32> to tensor<32x128xi32>
          %3 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {notile = #cinm.notile} : (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
          affine.yield %3 : tensor<1x128xi32>
        }
        %inserted_slice = tensor.insert_slice %2 into %arg3[0, %arg2] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<1x128xi32>
        affine.yield %inserted_slice : tensor<1x128xi32>
      }
      cinm.yield
    }
    return
  }
  func.func @mm_dimm4_opt(%arg0: tensor<16x64xi32>, %arg1: tensor<64x128xi32>) {
    cinm.compute attributes {workgroupShape = array<i64: 4, 128>} {
      %0 = tensor.empty() : tensor<16x128xi32>
      %1 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %0) -> (tensor<16x128xi32>) {
        %2 = affine.for %arg4 = 0 to 128 step 128 iter_args(%arg5 = %arg3) -> (tensor<16x128xi32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, 0] [1, 64] [1, 1] : tensor<16x64xi32> to tensor<1x64xi32>
          %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg4] [64, 128] [1, 1] : tensor<64x128xi32> to tensor<64x128xi32>
          %3 = tensor.empty() : tensor<1x128xi32>
          %4 = affine.for %arg6 = 0 to 128 step 128 iter_args(%arg7 = %3) -> (tensor<1x128xi32>) {
            %c0_i32 = arith.constant 0 : i32
            %splat = tensor.splat %c0_i32 : tensor<1x128xi32>
            %5 = affine.for %arg8 = 0 to 64 step 32 iter_args(%arg9 = %splat) -> (tensor<1x128xi32>) {
              %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg8] [1, 32] [1, 1] : tensor<1x64xi32> to tensor<1x32xi32>
              %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg6, %arg8] [32, 128] [1, 1] : tensor<64x128xi32> to tensor<32x128xi32>
              %6 = cinm.op.gemm %extracted_slice_2, %extracted_slice_3 plus %arg9 {notile = #cinm.notile} : (tensor<1x32xi32>, tensor<32x128xi32>) -> tensor<1x128xi32>
              affine.yield %6 : tensor<1x128xi32>
            }
            %inserted_slice_1 = tensor.insert_slice %5 into %arg7[0, %arg6] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<1x128xi32>
            affine.yield %inserted_slice_1 : tensor<1x128xi32>
          }
          %inserted_slice = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [1, 128] [1, 1] : tensor<1x128xi32> into tensor<16x128xi32>
          affine.yield %inserted_slice : tensor<16x128xi32>
        }
        affine.yield %2 : tensor<16x128xi32>
      }
      cinm.yield
    }
    return
  }
}
