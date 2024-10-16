; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_xf32_0 = private constant float 0.000000e+00, align 64
@__constant_1xi64 = private constant [1 x i64] [i64 131072], align 64
@__constant_1024x128xf32 = private constant [1024 x [128 x float]] zeroinitializer, align 64
@__constant_2xi64 = private constant [2 x i64] [i64 1024, i64 128], align 64
@__constant_131072xf32 = private constant [131072 x float] zeroinitializer, align 64
@__constant_xf32 = private constant float 0xFFF0000000000000, align 64
@dpu_program = private constant [8 x i8] c"softmax\00"
@dpu_program_0 = private constant [10 x i8] c"softmax_0\00"

declare ptr @malloc(i64)

declare void @upmemrt_dpu_free(ptr)

declare void @upmemrt_dpu_gather(ptr, ptr, i64, i64, i64, i64, i64, ptr)

declare void @upmemrt_dpu_launch(ptr)

define private i64 @scatter_map_0(i64 %0) {
  ret i64 0
}

declare void @upmemrt_dpu_scatter(ptr, ptr, i64, i64, i64, i64, i64, ptr)

define private i64 @scatter_map(i64 %0) {
  %2 = mul i64 %0, 512
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 256
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, 131072
  %10 = add i64 %2, %9
  %11 = srem i64 %0, 256
  %12 = icmp slt i64 %11, 0
  %13 = add i64 %11, 256
  %14 = select i1 %12, i64 %13, i64 %11
  %15 = icmp slt i64 %14, 0
  %16 = sub i64 -1, %14
  %17 = select i1 %15, i64 %16, i64 %14
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, 8192
  %22 = add i64 %10, %21
  %23 = sdiv i64 %5, 16
  %24 = sub i64 -1, %23
  %25 = select i1 %3, i64 %24, i64 %23
  %26 = mul i64 %25, -8192
  %27 = add i64 %22, %26
  ret i64 %27
}

declare ptr @upmemrt_dpu_alloc(i32, i32, ptr)

define ptr @softmax(ptr %0) {
  %2 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64), i64 64))
  %3 = ptrtoint ptr %2 to i64
  %4 = add i64 %3, 63
  %5 = urem i64 %4, 64
  %6 = sub i64 %4, %5
  %7 = inttoptr i64 %6 to ptr
  %8 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %9 = ptrtoint ptr %8 to i64
  %10 = add i64 %9, 63
  %11 = urem i64 %10, 64
  %12 = sub i64 %10, %11
  %13 = inttoptr i64 %12 to ptr
  br label %14

14:                                               ; preds = %29, %1
  %15 = phi i64 [ %32, %29 ], [ 0, %1 ]
  %16 = icmp slt i64 %15, 128
  br i1 %16, label %17, label %33

17:                                               ; preds = %14
  %18 = mul i64 %15, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %13, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %19

19:                                               ; preds = %22, %17
  %20 = phi i64 [ %28, %22 ], [ 0, %17 ]
  %21 = icmp slt i64 %20, 1024
  br i1 %21, label %22, label %29

22:                                               ; preds = %19
  %23 = add i64 %18, %20
  %24 = getelementptr float, ptr %0, i64 %23
  %25 = load float, ptr %24, align 4
  %26 = load float, ptr %13, align 4
  %27 = call float @llvm.maximum.f32(float %25, float %26)
  store float %27, ptr %13, align 4
  %28 = add i64 %20, 1
  br label %19

29:                                               ; preds = %19
  %30 = load float, ptr %13, align 4
  %31 = getelementptr float, ptr %7, i64 %15
  store float %30, ptr %31, align 4
  %32 = add i64 %15, 1
  br label %14

33:                                               ; preds = %14
  %34 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %35 = ptrtoint ptr %34 to i64
  %36 = add i64 %35, 63
  %37 = urem i64 %36, 64
  %38 = sub i64 %36, %37
  %39 = inttoptr i64 %38 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %39, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %40

40:                                               ; preds = %43, %33
  %41 = phi i64 [ %48, %43 ], [ 0, %33 ]
  %42 = icmp slt i64 %41, 128
  br i1 %42, label %43, label %49

43:                                               ; preds = %40
  %44 = getelementptr float, ptr %7, i64 %41
  %45 = load float, ptr %44, align 4
  %46 = load float, ptr %39, align 4
  %47 = call float @llvm.maximum.f32(float %45, float %46)
  store float %47, ptr %39, align 4
  %48 = add i64 %41, 1
  br label %40

49:                                               ; preds = %40
  %50 = load float, ptr %39, align 4
  %51 = call ptr @upmemrt_dpu_alloc(i32 4, i32 16, ptr @dpu_program)
  call void @upmemrt_dpu_scatter(ptr %51, ptr %0, i64 4, i64 131072, i64 128, i64 8192, i64 0, ptr @scatter_map)
  %52 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64), i64 64))
  %53 = ptrtoint ptr %52 to i64
  %54 = add i64 %53, 63
  %55 = urem i64 %54, 64
  %56 = sub i64 %54, %55
  %57 = inttoptr i64 %56 to ptr
  store float %50, ptr %57, align 4
  %58 = getelementptr float, ptr %57, i32 1
  store float %50, ptr %58, align 4
  %59 = getelementptr float, ptr %57, i32 2
  store float %50, ptr %59, align 4
  %60 = getelementptr float, ptr %57, i32 3
  store float %50, ptr %60, align 4
  %61 = getelementptr float, ptr %57, i32 4
  store float %50, ptr %61, align 4
  %62 = getelementptr float, ptr %57, i32 5
  store float %50, ptr %62, align 4
  %63 = getelementptr float, ptr %57, i32 6
  store float %50, ptr %63, align 4
  %64 = getelementptr float, ptr %57, i32 7
  store float %50, ptr %64, align 4
  %65 = getelementptr float, ptr %57, i32 8
  store float %50, ptr %65, align 4
  %66 = getelementptr float, ptr %57, i32 9
  store float %50, ptr %66, align 4
  %67 = getelementptr float, ptr %57, i32 10
  store float %50, ptr %67, align 4
  %68 = getelementptr float, ptr %57, i32 11
  store float %50, ptr %68, align 4
  %69 = getelementptr float, ptr %57, i32 12
  store float %50, ptr %69, align 4
  %70 = getelementptr float, ptr %57, i32 13
  store float %50, ptr %70, align 4
  %71 = getelementptr float, ptr %57, i32 14
  store float %50, ptr %71, align 4
  %72 = getelementptr float, ptr %57, i32 15
  store float %50, ptr %72, align 4
  call void @upmemrt_dpu_scatter(ptr %51, ptr %57, i64 4, i64 16, i64 0, i64 64, i64 8192, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %51, ptr @__constant_1024x128xf32, i64 4, i64 131072, i64 128, i64 8192, i64 8256, ptr @scatter_map)
  call void @upmemrt_dpu_launch(ptr %51)
  %73 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 131072) to i64), i64 64))
  %74 = ptrtoint ptr %73 to i64
  %75 = add i64 %74, 63
  %76 = urem i64 %75, 64
  %77 = sub i64 %75, %76
  %78 = inttoptr i64 %77 to ptr
  call void @upmemrt_dpu_gather(ptr %51, ptr %78, i64 4, i64 131072, i64 128, i64 8192, i64 8256, ptr @scatter_map)
  call void @upmemrt_dpu_free(ptr %51)
  %79 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 131072) to i64), i64 64))
  %80 = ptrtoint ptr %79 to i64
  %81 = add i64 %80, 63
  %82 = urem i64 %81, 64
  %83 = sub i64 %81, %82
  %84 = inttoptr i64 %83 to ptr
  br label %85

85:                                               ; preds = %88, %49
  %86 = phi i64 [ %93, %88 ], [ 0, %49 ]
  %87 = icmp slt i64 %86, 131072
  br i1 %87, label %88, label %94

88:                                               ; preds = %85
  %89 = getelementptr float, ptr %78, i64 %86
  %90 = load float, ptr %89, align 4
  %91 = call float @llvm.exp.f32(float %90)
  %92 = getelementptr float, ptr %84, i64 %86
  store float %91, ptr %92, align 4
  %93 = add i64 %86, 1
  br label %85

94:                                               ; preds = %85
  %95 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 128) to i64), i64 64))
  %96 = ptrtoint ptr %95 to i64
  %97 = add i64 %96, 63
  %98 = urem i64 %97, 64
  %99 = sub i64 %97, %98
  %100 = inttoptr i64 %99 to ptr
  %101 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %102 = ptrtoint ptr %101 to i64
  %103 = add i64 %102, 63
  %104 = urem i64 %103, 64
  %105 = sub i64 %103, %104
  %106 = inttoptr i64 %105 to ptr
  br label %107

107:                                              ; preds = %122, %94
  %108 = phi i64 [ %125, %122 ], [ 0, %94 ]
  %109 = icmp slt i64 %108, 128
  br i1 %109, label %110, label %126

110:                                              ; preds = %107
  %111 = mul i64 %108, 1024
  call void @llvm.memcpy.p0.p0.i64(ptr %106, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %112

112:                                              ; preds = %115, %110
  %113 = phi i64 [ %121, %115 ], [ 0, %110 ]
  %114 = icmp slt i64 %113, 1024
  br i1 %114, label %115, label %122

115:                                              ; preds = %112
  %116 = add i64 %111, %113
  %117 = getelementptr float, ptr %84, i64 %116
  %118 = load float, ptr %117, align 4
  %119 = load float, ptr %106, align 4
  %120 = fadd float %118, %119
  store float %120, ptr %106, align 4
  %121 = add i64 %113, 1
  br label %112

122:                                              ; preds = %112
  %123 = load float, ptr %106, align 4
  %124 = getelementptr float, ptr %100, i64 %108
  store float %123, ptr %124, align 4
  %125 = add i64 %108, 1
  br label %107

126:                                              ; preds = %107
  %127 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %128 = ptrtoint ptr %127 to i64
  %129 = add i64 %128, 63
  %130 = urem i64 %129, 64
  %131 = sub i64 %129, %130
  %132 = inttoptr i64 %131 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %132, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %133

133:                                              ; preds = %136, %126
  %134 = phi i64 [ %141, %136 ], [ 0, %126 ]
  %135 = icmp slt i64 %134, 128
  br i1 %135, label %136, label %142

136:                                              ; preds = %133
  %137 = getelementptr float, ptr %100, i64 %134
  %138 = load float, ptr %137, align 4
  %139 = load float, ptr %132, align 4
  %140 = fadd float %138, %139
  store float %140, ptr %132, align 4
  %141 = add i64 %134, 1
  br label %133

142:                                              ; preds = %133
  %143 = load float, ptr %132, align 4
  %144 = call ptr @upmemrt_dpu_alloc(i32 4, i32 16, ptr @dpu_program_0)
  call void @upmemrt_dpu_scatter(ptr %144, ptr %84, i64 4, i64 131072, i64 128, i64 8192, i64 0, ptr @scatter_map)
  %145 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64), i64 64))
  %146 = ptrtoint ptr %145 to i64
  %147 = add i64 %146, 63
  %148 = urem i64 %147, 64
  %149 = sub i64 %147, %148
  %150 = inttoptr i64 %149 to ptr
  store float %143, ptr %150, align 4
  %151 = getelementptr float, ptr %150, i32 1
  store float %143, ptr %151, align 4
  %152 = getelementptr float, ptr %150, i32 2
  store float %143, ptr %152, align 4
  %153 = getelementptr float, ptr %150, i32 3
  store float %143, ptr %153, align 4
  %154 = getelementptr float, ptr %150, i32 4
  store float %143, ptr %154, align 4
  %155 = getelementptr float, ptr %150, i32 5
  store float %143, ptr %155, align 4
  %156 = getelementptr float, ptr %150, i32 6
  store float %143, ptr %156, align 4
  %157 = getelementptr float, ptr %150, i32 7
  store float %143, ptr %157, align 4
  %158 = getelementptr float, ptr %150, i32 8
  store float %143, ptr %158, align 4
  %159 = getelementptr float, ptr %150, i32 9
  store float %143, ptr %159, align 4
  %160 = getelementptr float, ptr %150, i32 10
  store float %143, ptr %160, align 4
  %161 = getelementptr float, ptr %150, i32 11
  store float %143, ptr %161, align 4
  %162 = getelementptr float, ptr %150, i32 12
  store float %143, ptr %162, align 4
  %163 = getelementptr float, ptr %150, i32 13
  store float %143, ptr %163, align 4
  %164 = getelementptr float, ptr %150, i32 14
  store float %143, ptr %164, align 4
  %165 = getelementptr float, ptr %150, i32 15
  store float %143, ptr %165, align 4
  call void @upmemrt_dpu_scatter(ptr %144, ptr %150, i64 4, i64 16, i64 0, i64 64, i64 8192, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %144, ptr @__constant_1024x128xf32, i64 4, i64 131072, i64 128, i64 8192, i64 8256, ptr @scatter_map)
  call void @upmemrt_dpu_launch(ptr %144)
  %166 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 131072) to i64), i64 64))
  %167 = ptrtoint ptr %166 to i64
  %168 = add i64 %167, 63
  %169 = urem i64 %168, 64
  %170 = sub i64 %168, %169
  %171 = inttoptr i64 %170 to ptr
  call void @upmemrt_dpu_gather(ptr %144, ptr %171, i64 4, i64 131072, i64 128, i64 8192, i64 8256, ptr @scatter_map)
  call void @upmemrt_dpu_free(ptr %144)
  ret ptr %166
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
