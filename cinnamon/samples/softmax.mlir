func.func @softmax(%vec : tensor<?xf32>, %1 : index) -> tensor<1024xf32> {
	%r = cinm.compute -> tensor<1024xf32> {
		%zero = arith.constant dense<0.0> : tensor<f32>
		%ninft = linalg.log ins(%zero : tensor<f32>) outs(%zero : tensor<f32>) -> tensor<f32>
		%ninf = tensor.extract %ninft [] : tensor<f32>
		%padded = tensor.pad %vec low[%1] high[1024] {
		^bb0(%arg1: index):
			tensor.yield %ninf : f32
		} : tensor<?xf32> to tensor<1024xf32>

		%max = cinm.op.reduce max (%padded) : tensor<1024xf32>
		%t = cinm.op.subs %padded, %max : tensor<1024xf32>
		%shape = tensor.empty() : tensor<1024xf32>
		%e = linalg.exp ins(%t : tensor<1024xf32>) outs(%shape : tensor<1024xf32>) -> tensor<1024xf32>
		%s = cinm.op.reduce add (%e) : tensor<1024xf32>
		%r = cinm.op.divs %e, %s : tensor<1024xf32>
		cinm.yield %r : tensor<1024xf32>
	}

	return %r : tensor<1024xf32>
}
