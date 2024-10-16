; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1x512xi32 = private constant [1 x [512 x i32]] zeroinitializer, align 64
@dpu_program = private constant [5 x i8] c"main\00"

declare ptr @malloc(i64)

declare void @upmemrt_dpu_free(ptr)

declare void @upmemrt_dpu_gather(ptr, ptr, i64, i64, i64, i64, i64, ptr)

declare void @upmemrt_dpu_launch(ptr)

define private i64 @scatter_map_1(i64 %0) {
  %2 = mul i64 %0, 4
  %3 = srem i64 %0, 256
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 256
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 64
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -64
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_0(i64 %0) {
  %2 = mul i64 %0, 1024
  %3 = srem i64 %0, 256
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 256
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 16384
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -16384
  %22 = add i64 %14, %21
  ret i64 %22
}

declare void @upmemrt_dpu_scatter(ptr, ptr, i64, i64, i64, i64, i64, ptr)

define private i64 @scatter_map(i64 %0) {
  ret i64 0
}

declare ptr @upmemrt_dpu_alloc(i32, i32, ptr)

define void @main() {
  %1 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1048576) to i64), i64 64))
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1048576) to i64), i64 64))
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 63
  %10 = urem i64 %9, 64
  %11 = sub i64 %9, %10
  %12 = inttoptr i64 %11 to ptr
  %13 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1048576) to i64), i64 64))
  %14 = ptrtoint ptr %13 to i64
  %15 = add i64 %14, 63
  %16 = urem i64 %15, 64
  %17 = sub i64 %15, %16
  %18 = inttoptr i64 %17 to ptr
  %19 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1048576) to i64), i64 64))
  %20 = ptrtoint ptr %19 to i64
  %21 = add i64 %20, 63
  %22 = urem i64 %21, 64
  %23 = sub i64 %21, %22
  %24 = inttoptr i64 %23 to ptr
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %19, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, ptr %24, 1
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 0, 2
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 1024, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 1024, 3, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 1024, 4, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 1, 4, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %24, ptr %18, i64 mul (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 1048576), i1 false)
  %32 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 512) to i64), i64 64))
  %33 = ptrtoint ptr %32 to i64
  %34 = add i64 %33, 63
  %35 = urem i64 %34, 64
  %36 = sub i64 %34, %35
  %37 = inttoptr i64 %36 to ptr
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %32, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, ptr %37, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 0, 2
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 1, 3, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 512, 3, 1
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 512, 4, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 1, 4, 1
  %45 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 131072) to i64), i64 64))
  %46 = ptrtoint ptr %45 to i64
  %47 = add i64 %46, 63
  %48 = urem i64 %47, 64
  %49 = sub i64 %47, %48
  %50 = inttoptr i64 %49 to ptr
  br label %51

51:                                               ; preds = %163, %0
  %52 = phi i64 [ %164, %163 ], [ 0, %0 ]
  %53 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %79, %163 ], [ %31, %0 ]
  %54 = icmp slt i64 %52, 1024
  br i1 %54, label %55, label %165

55:                                               ; preds = %51
  %56 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1048576) to i64), i64 64))
  %57 = ptrtoint ptr %56 to i64
  %58 = add i64 %57, 63
  %59 = urem i64 %58, 64
  %60 = sub i64 %58, %59
  %61 = inttoptr i64 %60 to ptr
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %56, 0
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, ptr %61, 1
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 0, 2
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 1024, 3, 0
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 1024, 3, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, i64 1024, 4, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, i64 1, 4, 1
  %69 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 3, 0
  %70 = mul i64 %69, 1
  %71 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 3, 1
  %72 = mul i64 %70, %71
  %73 = mul i64 %72, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %74 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 2
  %76 = getelementptr i32, ptr %74, i64 %75
  call void @llvm.memcpy.p0.p0.i64(ptr %61, ptr %76, i64 %73, i1 false)
  br label %77

77:                                               ; preds = %129, %55
  %78 = phi i64 [ %162, %129 ], [ 0, %55 ]
  %79 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %142, %129 ], [ %68, %55 ]
  %80 = icmp slt i64 %78, 1024
  br i1 %80, label %81, label %163

81:                                               ; preds = %77
  call void @llvm.memcpy.p0.p0.i64(ptr %37, ptr @__constant_1x512xi32, i64 mul (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64), i64 512), i1 false)
  br label %82

82:                                               ; preds = %110, %81
  %83 = phi i64 [ %128, %110 ], [ 0, %81 ]
  %84 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %127, %110 ], [ %44, %81 ]
  %85 = icmp slt i64 %83, 1024
  br i1 %85, label %86, label %129

86:                                               ; preds = %82
  %87 = mul i64 %52, 1024
  %88 = add i64 %87, %83
  %89 = call ptr @upmemrt_dpu_alloc(i32 2, i32 16, ptr @dpu_program)
  br label %90

90:                                               ; preds = %108, %86
  %91 = phi i64 [ %109, %108 ], [ 0, %86 ]
  %92 = icmp slt i64 %91, 512
  br i1 %92, label %93, label %110

93:                                               ; preds = %90
  br label %94

94:                                               ; preds = %97, %93
  %95 = phi i64 [ %107, %97 ], [ 0, %93 ]
  %96 = icmp slt i64 %95, 256
  br i1 %96, label %97, label %108

97:                                               ; preds = %94
  %98 = add i64 %83, %95
  %99 = add i64 %78, %91
  %100 = mul i64 %98, 1024
  %101 = add i64 %100, %99
  %102 = getelementptr i32, ptr %12, i64 %101
  %103 = load i32, ptr %102, align 4
  %104 = mul i64 %91, 256
  %105 = add i64 %104, %95
  %106 = getelementptr i32, ptr %50, i64 %105
  store i32 %103, ptr %106, align 4
  %107 = add i64 %95, 1
  br label %94

108:                                              ; preds = %94
  %109 = add i64 %91, 1
  br label %90

110:                                              ; preds = %90
  %111 = getelementptr inbounds i32, ptr %6, i64 %88
  call void @upmemrt_dpu_scatter(ptr %89, ptr %111, i64 4, i64 256, i64 0, i64 16384, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %89, ptr %50, i64 4, i64 131072, i64 256, i64 16384, i64 16384, ptr @scatter_map_0)
  %112 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, 1
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, 2
  %114 = getelementptr inbounds i32, ptr %112, i64 %113
  call void @upmemrt_dpu_scatter(ptr %89, ptr %114, i64 4, i64 512, i64 1, i64 64, i64 32768, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %89)
  %115 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 512) to i64), i64 64))
  %116 = ptrtoint ptr %115 to i64
  %117 = add i64 %116, 63
  %118 = urem i64 %117, 64
  %119 = sub i64 %117, %118
  %120 = inttoptr i64 %119 to ptr
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %115, 0
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, ptr %120, 1
  %123 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %122, i64 0, 2
  %124 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %123, i64 1, 3, 0
  %125 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %124, i64 512, 3, 1
  %126 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %125, i64 512, 4, 0
  %127 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %126, i64 1, 4, 1
  call void @upmemrt_dpu_gather(ptr %89, ptr %120, i64 4, i64 512, i64 1, i64 64, i64 32768, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %89)
  %128 = add i64 %83, 256
  br label %82

129:                                              ; preds = %82
  %130 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr null, i32 1048576) to i64), i64 64))
  %131 = ptrtoint ptr %130 to i64
  %132 = add i64 %131, 63
  %133 = urem i64 %132, 64
  %134 = sub i64 %132, %133
  %135 = inttoptr i64 %134 to ptr
  %136 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %130, 0
  %137 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %136, ptr %135, 1
  %138 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %137, i64 0, 2
  %139 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %138, i64 1024, 3, 0
  %140 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %139, i64 1024, 3, 1
  %141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %140, i64 1024, 4, 0
  %142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %141, i64 1, 4, 1
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, 3, 0
  %144 = mul i64 %143, 1
  %145 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, 3, 1
  %146 = mul i64 %144, %145
  %147 = mul i64 %146, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %148 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, 1
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, 2
  %150 = getelementptr i32, ptr %148, i64 %149
  call void @llvm.memcpy.p0.p0.i64(ptr %135, ptr %150, i64 %147, i1 false)
  %151 = mul i64 %52, 1024
  %152 = add i64 %151, %78
  %153 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, 3, 0
  %154 = mul i64 %153, 1
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, 3, 1
  %156 = mul i64 %154, %155
  %157 = mul i64 %156, ptrtoint (ptr getelementptr (i32, ptr null, i32 1) to i64)
  %158 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, 1
  %159 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %84, 2
  %160 = getelementptr i32, ptr %158, i64 %159
  %161 = getelementptr i32, ptr %135, i64 %152
  call void @llvm.memcpy.p0.p0.i64(ptr %161, ptr %160, i64 %157, i1 false)
  %162 = add i64 %78, 512
  br label %77

163:                                              ; preds = %77
  %164 = add i64 %52, 1
  br label %51

165:                                              ; preds = %51
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
