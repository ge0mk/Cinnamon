#map = affine_map<(d0) -> (d0 * 4 + ((d0 mod 256) floordiv 16) * 64 - (d0 floordiv 16) * 64)>
#map1 = affine_map<(d0) -> (d0 * 1024 + ((d0 mod 256) floordiv 16) * 16384 - (d0 floordiv 16) * 16384)>
#map2 = affine_map<(d0) -> (0)>
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @upmemrt_dpu_free(!llvm.ptr)
  llvm.func @upmemrt_dpu_gather(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr)
  llvm.func @upmemrt_dpu_launch(!llvm.ptr)
  llvm.func private @scatter_map_1(%arg0: i64) -> i64 attributes {upmem.generated_from = #map} {
    %0 = llvm.mlir.constant(-64 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(-1 : index) : i64
    %3 = llvm.mlir.constant(16 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(256 : index) : i64
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mul %arg0, %6  : i64
    %8 = llvm.srem %arg0, %5  : i64
    %9 = llvm.icmp "slt" %8, %4 : i64
    %10 = llvm.add %8, %5  : i64
    %11 = llvm.select %9, %10, %8 : i1, i64
    %12 = llvm.icmp "slt" %11, %4 : i64
    %13 = llvm.sub %2, %11  : i64
    %14 = llvm.select %12, %13, %11 : i1, i64
    %15 = llvm.sdiv %14, %3  : i64
    %16 = llvm.sub %2, %15  : i64
    %17 = llvm.select %12, %16, %15 : i1, i64
    %18 = llvm.mul %17, %1  : i64
    %19 = llvm.add %7, %18  : i64
    %20 = llvm.icmp "slt" %arg0, %4 : i64
    %21 = llvm.sub %2, %arg0  : i64
    %22 = llvm.select %20, %21, %arg0 : i1, i64
    %23 = llvm.sdiv %22, %3  : i64
    %24 = llvm.sub %2, %23  : i64
    %25 = llvm.select %20, %24, %23 : i1, i64
    %26 = llvm.mul %25, %0  : i64
    %27 = llvm.add %19, %26  : i64
    llvm.return %27 : i64
  }
  llvm.func private @scatter_map_0(%arg0: i64) -> i64 attributes {upmem.generated_from = #map1} {
    %0 = llvm.mlir.constant(-16384 : index) : i64
    %1 = llvm.mlir.constant(16384 : index) : i64
    %2 = llvm.mlir.constant(-1 : index) : i64
    %3 = llvm.mlir.constant(16 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(256 : index) : i64
    %6 = llvm.mlir.constant(1024 : index) : i64
    %7 = llvm.mul %arg0, %6  : i64
    %8 = llvm.srem %arg0, %5  : i64
    %9 = llvm.icmp "slt" %8, %4 : i64
    %10 = llvm.add %8, %5  : i64
    %11 = llvm.select %9, %10, %8 : i1, i64
    %12 = llvm.icmp "slt" %11, %4 : i64
    %13 = llvm.sub %2, %11  : i64
    %14 = llvm.select %12, %13, %11 : i1, i64
    %15 = llvm.sdiv %14, %3  : i64
    %16 = llvm.sub %2, %15  : i64
    %17 = llvm.select %12, %16, %15 : i1, i64
    %18 = llvm.mul %17, %1  : i64
    %19 = llvm.add %7, %18  : i64
    %20 = llvm.icmp "slt" %arg0, %4 : i64
    %21 = llvm.sub %2, %arg0  : i64
    %22 = llvm.select %20, %21, %arg0 : i1, i64
    %23 = llvm.sdiv %22, %3  : i64
    %24 = llvm.sub %2, %23  : i64
    %25 = llvm.select %20, %24, %23 : i1, i64
    %26 = llvm.mul %25, %0  : i64
    %27 = llvm.add %19, %26  : i64
    llvm.return %27 : i64
  }
  llvm.func @upmemrt_dpu_scatter(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr)
  llvm.func private @scatter_map(%arg0: i64) -> i64 attributes {upmem.generated_from = #map2} {
    %0 = llvm.mlir.constant(0 : index) : i64
    llvm.return %0 : i64
  }
  llvm.func @upmemrt_dpu_alloc(i32, i32, !llvm.ptr) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x512xi32(dense<0> : tensor<1x512xi32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<512 x i32>>
  llvm.func @main() {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(64 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(512 : i64) : i64
    %4 = llvm.mlir.constant(32768 : i64) : i64
    %5 = llvm.mlir.constant(131072 : i64) : i64
    %6 = llvm.mlir.constant(16384 : i64) : i64
    %7 = llvm.mlir.constant(256 : i64) : i64
    %8 = llvm.mlir.constant(4 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(16 : i32) : i32
    %11 = llvm.mlir.constant(2 : i32) : i32
    %12 = llvm.mlir.constant(256 : index) : i64
    %13 = llvm.mlir.constant(512 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(1024 : index) : i64
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[1048576] : (!llvm.ptr) -> !llvm.ptr, i32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.add %19, %0  : i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.sub %0, %14  : i64
    %24 = llvm.add %22, %23  : i64
    %25 = llvm.urem %24, %0  : i64
    %26 = llvm.sub %24, %25  : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr
    %28 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.add %30, %23  : i64
    %32 = llvm.urem %31, %0  : i64
    %33 = llvm.sub %31, %32  : i64
    %34 = llvm.inttoptr %33 : i64 to !llvm.ptr
    %35 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.add %36, %23  : i64
    %38 = llvm.urem %37, %0  : i64
    %39 = llvm.sub %37, %38  : i64
    %40 = llvm.inttoptr %39 : i64 to !llvm.ptr
    %41 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.add %42, %23  : i64
    %44 = llvm.urem %43, %0  : i64
    %45 = llvm.sub %43, %44  : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    %47 = llvm.insertvalue %41, %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.insertvalue %46, %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.insertvalue %16, %48[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %15, %49[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %15, %50[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %15, %51[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %14, %52[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mul %15, %14  : i64
    %55 = llvm.mul %54, %15  : i64
    %56 = llvm.getelementptr %17[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %57 = llvm.ptrtoint %56 : !llvm.ptr to i64
    %58 = llvm.mul %55, %57  : i64
    "llvm.intr.memcpy"(%46, %40, %58) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %59 = llvm.getelementptr %17[512] : (!llvm.ptr) -> !llvm.ptr, i32
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.add %60, %0  : i64
    %62 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr
    %63 = llvm.ptrtoint %62 : !llvm.ptr to i64
    %64 = llvm.add %63, %23  : i64
    %65 = llvm.urem %64, %0  : i64
    %66 = llvm.sub %64, %65  : i64
    %67 = llvm.inttoptr %66 : i64 to !llvm.ptr
    %68 = llvm.insertvalue %62, %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %67, %68[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %16, %69[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %14, %70[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %13, %71[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %13, %72[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.insertvalue %14, %73[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.getelementptr %17[131072] : (!llvm.ptr) -> !llvm.ptr, i32
    %76 = llvm.ptrtoint %75 : !llvm.ptr to i64
    %77 = llvm.add %76, %0  : i64
    %78 = llvm.call @malloc(%77) : (i64) -> !llvm.ptr
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.add %79, %23  : i64
    %81 = llvm.urem %80, %0  : i64
    %82 = llvm.sub %80, %81  : i64
    %83 = llvm.inttoptr %82 : i64 to !llvm.ptr
    llvm.br ^bb1(%16, %53 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb1(%84: i64, %85: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb0, ^bb14
    %86 = llvm.icmp "slt" %84, %15 : i64
    llvm.cond_br %86, ^bb2, ^bb15
  ^bb2:  // pred: ^bb1
    %87 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %88 = llvm.ptrtoint %87 : !llvm.ptr to i64
    %89 = llvm.add %88, %23  : i64
    %90 = llvm.urem %89, %0  : i64
    %91 = llvm.sub %89, %90  : i64
    %92 = llvm.inttoptr %91 : i64 to !llvm.ptr
    %93 = llvm.insertvalue %87, %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %92, %93[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.insertvalue %16, %94[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.insertvalue %15, %95[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.insertvalue %15, %96[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.insertvalue %15, %97[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %14, %98[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mul %100, %14  : i64
    %102 = llvm.extractvalue %85[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mul %101, %102  : i64
    %104 = llvm.mul %103, %57  : i64
    %105 = llvm.extractvalue %85[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.extractvalue %85[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.getelementptr %105[%106] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%92, %107, %104) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%16, %99 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb3(%108: i64, %109: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb2, ^bb13
    %110 = llvm.icmp "slt" %108, %15 : i64
    llvm.cond_br %110, ^bb4, ^bb14
  ^bb4:  // pred: ^bb3
    %111 = llvm.mlir.addressof @__constant_1x512xi32 : !llvm.ptr
    %112 = llvm.getelementptr %111[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<512 x i32>>
    %113 = llvm.mul %14, %14  : i64
    %114 = llvm.mul %113, %13  : i64
    %115 = llvm.mul %114, %57  : i64
    "llvm.intr.memcpy"(%67, %112, %115) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb5(%16, %74 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb5(%116: i64, %117: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb4, ^bb12
    %118 = llvm.icmp "slt" %116, %15 : i64
    llvm.cond_br %118, ^bb6, ^bb13
  ^bb6:  // pred: ^bb5
    %119 = llvm.mul %84, %15  : i64
    %120 = llvm.add %119, %116  : i64
    %121 = llvm.mlir.addressof @dpu_program : !llvm.ptr
    %122 = llvm.call @upmemrt_dpu_alloc(%11, %10, %121) : (i32, i32, !llvm.ptr) -> !llvm.ptr
    llvm.br ^bb7(%16 : i64)
  ^bb7(%123: i64):  // 2 preds: ^bb6, ^bb11
    %124 = llvm.icmp "slt" %123, %13 : i64
    llvm.cond_br %124, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%16 : i64)
  ^bb9(%125: i64):  // 2 preds: ^bb8, ^bb10
    %126 = llvm.icmp "slt" %125, %12 : i64
    llvm.cond_br %126, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %127 = llvm.add %116, %125  : i64
    %128 = llvm.add %108, %123  : i64
    %129 = llvm.mul %127, %15  : i64
    %130 = llvm.add %129, %128  : i64
    %131 = llvm.getelementptr %34[%130] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %132 = llvm.load %131 : !llvm.ptr -> i32
    %133 = llvm.mul %123, %12  : i64
    %134 = llvm.add %133, %125  : i64
    %135 = llvm.getelementptr %83[%134] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %132, %135 : i32, !llvm.ptr
    %136 = llvm.add %125, %14  : i64
    llvm.br ^bb9(%136 : i64)
  ^bb11:  // pred: ^bb9
    %137 = llvm.add %123, %14  : i64
    llvm.br ^bb7(%137 : i64)
  ^bb12:  // pred: ^bb7
    %138 = llvm.mlir.addressof @scatter_map : !llvm.ptr
    %139 = llvm.getelementptr inbounds %27[%120] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.call @upmemrt_dpu_scatter(%122, %139, %8, %7, %9, %6, %9, %138) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> ()
    %140 = llvm.mlir.addressof @scatter_map_0 : !llvm.ptr
    llvm.call @upmemrt_dpu_scatter(%122, %83, %8, %5, %7, %6, %6, %140) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> ()
    %141 = llvm.mlir.addressof @scatter_map_1 : !llvm.ptr
    %142 = llvm.extractvalue %117[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.extractvalue %117[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.getelementptr inbounds %142[%143] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.call @upmemrt_dpu_scatter(%122, %144, %8, %3, %2, %1, %4, %141) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> ()
    llvm.call @upmemrt_dpu_launch(%122) : (!llvm.ptr) -> ()
    %145 = llvm.call @malloc(%61) : (i64) -> !llvm.ptr
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.add %146, %23  : i64
    %148 = llvm.urem %147, %0  : i64
    %149 = llvm.sub %147, %148  : i64
    %150 = llvm.inttoptr %149 : i64 to !llvm.ptr
    %151 = llvm.insertvalue %145, %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.insertvalue %150, %151[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %16, %152[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.insertvalue %14, %153[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.insertvalue %13, %154[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.insertvalue %13, %155[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %157 = llvm.insertvalue %14, %156[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @upmemrt_dpu_gather(%122, %150, %8, %3, %2, %1, %4, %141) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> ()
    llvm.call @upmemrt_dpu_free(%122) : (!llvm.ptr) -> ()
    %158 = llvm.add %116, %12  : i64
    llvm.br ^bb5(%158, %157 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb13:  // pred: ^bb5
    %159 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %160 = llvm.ptrtoint %159 : !llvm.ptr to i64
    %161 = llvm.add %160, %23  : i64
    %162 = llvm.urem %161, %0  : i64
    %163 = llvm.sub %161, %162  : i64
    %164 = llvm.inttoptr %163 : i64 to !llvm.ptr
    %165 = llvm.insertvalue %159, %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %164, %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.insertvalue %16, %166[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.insertvalue %15, %167[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.insertvalue %15, %168[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.insertvalue %15, %169[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %171 = llvm.insertvalue %14, %170[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.extractvalue %109[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %173 = llvm.mul %172, %14  : i64
    %174 = llvm.extractvalue %109[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.mul %173, %174  : i64
    %176 = llvm.mul %175, %57  : i64
    %177 = llvm.extractvalue %109[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.extractvalue %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %179 = llvm.getelementptr %177[%178] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%164, %179, %176) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %180 = llvm.mul %84, %15  : i64
    %181 = llvm.add %180, %108  : i64
    %182 = llvm.extractvalue %117[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.mul %182, %14  : i64
    %184 = llvm.extractvalue %117[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %185 = llvm.mul %183, %184  : i64
    %186 = llvm.mul %185, %57  : i64
    %187 = llvm.extractvalue %117[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %188 = llvm.extractvalue %117[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %189 = llvm.getelementptr %187[%188] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %190 = llvm.getelementptr %164[%181] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    "llvm.intr.memcpy"(%190, %189, %186) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %191 = llvm.add %108, %13  : i64
    llvm.br ^bb3(%191, %171 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb14:  // pred: ^bb3
    %192 = llvm.add %84, %14  : i64
    llvm.br ^bb1(%192, %109 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb15:  // pred: ^bb1
    llvm.return
  }
  llvm.mlir.global private constant @dpu_program("main\00") {addr_space = 0 : i32}
}

