module {
  func.func @main() {
    %0 = tensor.empty() : tensor<1024x1024xi32>
    %1 = tensor.empty() : tensor<1024x1024xi32>
    %2 = cinm.compute attributes {workgroupShape = array<i64: 2, 16, 16>} -> tensor<1024x1024xi32> {
      %3 = tensor.empty() : tensor<1024x1024xi32>
      %4 = affine.for %arg0 = 0 to 1024 iter_args(%arg1 = %3) -> (tensor<1024x1024xi32>) {
        %5 = affine.for %arg2 = 0 to 1024 step 512 iter_args(%arg3 = %arg1) -> (tensor<1024x1024xi32>) {
          %cst = arith.constant dense<0> : tensor<1x512xi32>
          %6 = affine.for %arg4 = 0 to 1024 step 256 iter_args(%arg5 = %cst) -> (tensor<1x512xi32>) {
            %extracted_slice = tensor.extract_slice %0[%arg0, %arg4] [1, 256] [1, 1] : tensor<1024x1024xi32> to tensor<1x256xi32>
            %extracted_slice_0 = tensor.extract_slice %1[%arg4, %arg2] [256, 512] [1, 1] : tensor<1024x1024xi32> to tensor<256x512xi32>
            %7 = cinm.op.gemm %extracted_slice, %extracted_slice_0 plus %arg5 {cinm.notile} : (tensor<1x256xi32>, tensor<256x512xi32>) -> tensor<1x512xi32>
            affine.yield %7 : tensor<1x512xi32>
          }
          %inserted_slice = tensor.insert_slice %6 into %arg3[%arg0, %arg2] [1, 512] [1, 1] : tensor<1x512xi32> into tensor<1024x1024xi32>
          affine.yield %inserted_slice : tensor<1024x1024xi32>
        }
        affine.yield %5 : tensor<1024x1024xi32>
      }
      cinm.yield %4 : tensor<1024x1024xi32>
    }
    return
  }
}

