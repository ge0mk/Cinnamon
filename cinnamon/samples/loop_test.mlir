module {
  func.func @foo() -> tensor<64x128x32xi64> {
    %z = arith.constant 0 : i64
    %t1 = tensor.empty() : tensor<64x128x32xi64>

    %r = affine.for %i = 0 to 64 iter_args(%t2 = %t1) -> (tensor<64x128x32xi64>) {
      %r1 = affine.for %k = 0 to 128 iter_args(%t3 = %t2) -> (tensor<64x128x32xi64>) {
      	%r2 = affine.for %l = 0 to 32 iter_args(%t4 = %t3) -> (tensor<64x128x32xi64>) {
        	%t5 = tensor.insert %z into %t4[%i, %k, %l] : tensor<64x128x32xi64>
        	affine.yield %t5 : tensor<64x128x32xi64>
        }
        affine.yield %r2 : tensor<64x128x32xi64>
      }
      affine.yield %r1 : tensor<64x128x32xi64>
    }
    return %r : tensor<64x128x32xi64>
  }

  func.func @bar() -> tensor<64x128xi64> {
    %z = arith.constant 0 : i64
    %t1 = tensor.empty() : tensor<64x128xi64>

    %r1 = affine.for %i = 0 to 64 iter_args(%t2 = %t1) -> (tensor<64x128xi64>) {
      %r1 = tensor.insert %z into %t2[%i, %i] : tensor<64x128xi64>
      affine.yield %r1 : tensor<64x128xi64>
    }

    %r2 = affine.for %i = 0 to 64 iter_args(%t2 = %t1) -> (tensor<64x128xi64>) {
      %r2 = tensor.insert %z into %t2[%i, %i] : tensor<64x128xi64>
      affine.yield %r2 : tensor<64x128xi64>
    }

    %r3 = affine.for %i = 0 to 64 iter_args(%t2 = %t1) -> (tensor<64x128xi64>) {
      %r3 = tensor.insert %z into %t2[%i, %i] : tensor<64x128xi64>
      affine.yield %r3 : tensor<64x128xi64>
    }

    return %r3 : tensor<64x128xi64>
  }
}
