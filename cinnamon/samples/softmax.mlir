func.func @softmax(%vec : tensor<131072xf32>) -> tensor<131072xf32> {
	%r = cinm.compute attributes { workgroupShape = array<i64: 4,16,16> } -> tensor<131072xf32> {
		%max = cinm.op.reduce max (%vec) : tensor<131072xf32>
		%t = cinm.op.subs %vec, %max : tensor<131072xf32>
		%shape = tensor.empty() : tensor<131072xf32>
		%e = linalg.exp ins(%t : tensor<131072xf32>) outs(%shape : tensor<131072xf32>) -> tensor<131072xf32>
		%s = cinm.op.reduce add (%e) : tensor<131072xf32>
		%r = cinm.op.divs %e, %s : tensor<131072xf32>
		cinm.yield %r : tensor<131072xf32>
	}

	return %r : tensor<131072xf32>
}
