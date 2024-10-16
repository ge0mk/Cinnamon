func.func @main() {
	%a = tensor.empty() : tensor<1024x1024xi32>
	%b = tensor.empty() : tensor<1024x1024xi32>
	%res = cinm.compute attributes{ workgroupShape = array<i64: 2, 16, 16> }-> tensor<1024x1024xi32> {
		%d = cinm.op.gemm %a, %b : (tensor<1024x1024xi32>, tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
		cinm.yield %d : tensor<1024x1024xi32>
	}
	return
}
