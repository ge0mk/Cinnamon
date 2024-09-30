; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@__constant_1xi64_3 = private constant [1 x i64] [i64 32768], align 64
@__constant_256x1xf32 = private constant [256 x [1 x float]] zeroinitializer, align 64
@__constant_2xi64_4 = private constant [2 x i64] [i64 768, i64 1], align 64
@__constant_1xi64_2 = private constant [1 x i64] [i64 768], align 64
@__constant_48x6xf32 = private constant [48 x [6 x float]] zeroinitializer, align 64
@__constant_2xi64_3 = private constant [2 x i64] [i64 48, i64 6], align 64
@__constant_48x1xf32 = private constant [48 x [1 x float]] zeroinitializer, align 64
@__constant_2xi64_2 = private constant [2 x i64] [i64 288, i64 1], align 64
@__constant_1xi64_1 = private constant [1 x i64] [i64 288], align 64
@__constant_16x18xf32 = private constant [16 x [18 x float]] zeroinitializer, align 64
@__constant_2xi64_1 = private constant [2 x i64] [i64 16, i64 18], align 64
@__constant_288xf32 = private constant [288 x float] zeroinitializer, align 64
@__constant_1xi64_0 = private constant [1 x i64] [i64 48], align 64
@__constant_8x6xf32 = private constant [8 x [6 x float]] zeroinitializer, align 64
@__constant_2xi64_0 = private constant [2 x i64] [i64 8, i64 6], align 64
@__constant_48xf32 = private constant [48 x float] zeroinitializer, align 64
@__constant_xf32_0 = private constant float 0.000000e+00, align 64
@__constant_1xi64 = private constant [1 x i64] [i64 256], align 64
@__constant_128x2xf32 = private constant [128 x [2 x float]] zeroinitializer, align 64
@__constant_2xi64 = private constant [2 x i64] [i64 128, i64 2], align 64
@__constant_256xf32 = private constant [256 x float] zeroinitializer, align 64
@__constant_xf32 = private constant float 0xFFF0000000000000, align 64
@dpu_program = private constant [8 x i8] c"forward\00"
@dpu_program_0 = private constant [8 x i8] c"forward\00"
@dpu_program_1 = private constant [8 x i8] c"forward\00"
@dpu_program_2 = private constant [8 x i8] c"forward\00"
@dpu_program_3 = private constant [10 x i8] c"forward_3\00"
@dpu_program_4 = private constant [8 x i8] c"forward\00"
@dpu_program_5 = private constant [8 x i8] c"forward\00"
@dpu_program_6 = private constant [10 x i8] c"forward_6\00"
@dpu_program_7 = private constant [10 x i8] c"forward_3\00"
@dpu_program_8 = private constant [10 x i8] c"forward_8\00"
@dpu_program_9 = private constant [5 x i8] c"attn\00"
@dpu_program_10 = private constant [7 x i8] c"attn_9\00"
@dpu_program_11 = private constant [10 x i8] c"forward_3\00"
@dpu_program_12 = private constant [8 x i8] c"rmsnorm\00"
@dpu_program_13 = private constant [11 x i8] c"rmsnorm_11\00"
@dpu_program_14 = private constant [8 x i8] c"rmsnorm\00"
@dpu_program_15 = private constant [8 x i8] c"softmax\00"
@dpu_program_16 = private constant [11 x i8] c"softmax_13\00"

declare void @memrefCopy(i64, ptr, ptr)

declare ptr @malloc(i64)

define private i64 @scatter_map_8(i64 %0) {
  %2 = mul i64 %0, 8
  %3 = srem i64 %0, 128
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 128
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 128
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -128
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_7(i64 %0) {
  %2 = mul i64 %0, 72
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 16
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, -1152
  %10 = add i64 %2, %9
  ret i64 %10
}

define private i64 @scatter_map_6(i64 %0) {
  %2 = mul i64 %0, 24
  %3 = icmp slt i64 %0, 0
  %4 = sub i64 -1, %0
  %5 = select i1 %3, i64 %4, i64 %0
  %6 = sdiv i64 %5, 8
  %7 = sub i64 -1, %6
  %8 = select i1 %3, i64 %7, i64 %6
  %9 = mul i64 %8, -192
  %10 = add i64 %2, %9
  ret i64 %10
}

define private i64 @scatter_map_5(i64 %0) {
  %2 = mul i64 %0, 4
  %3 = srem i64 %0, 128
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 128
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

define private i64 @scatter_map_4(i64 %0) {
  %2 = mul i64 %0, 1152
  %3 = srem i64 %0, 128
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 128
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 16
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 18432
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 16
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -18432
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_3(i64 %0) {
  %2 = mul i64 %0, 3072
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 24576
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -24576
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_2(i64 %0) {
  %2 = mul i64 %0, 24
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 192
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -192
  %22 = add i64 %14, %21
  ret i64 %22
}

declare void @upmemrt_dpu_free(ptr)

declare void @upmemrt_dpu_gather(ptr, ptr, i64, i64, i64, i64, i64, ptr)

declare void @upmemrt_dpu_launch(ptr)

define private i64 @scatter_map_1(i64 %0) {
  %2 = mul i64 %0, 4
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 32
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -32
  %22 = add i64 %14, %21
  ret i64 %22
}

define private i64 @scatter_map_0(i64 %0) {
  ret i64 0
}

declare void @upmemrt_dpu_scatter(ptr, ptr, i64, i64, i64, i64, i64, ptr)

define private i64 @scatter_map(i64 %0) {
  %2 = mul i64 %0, 1152
  %3 = srem i64 %0, 48
  %4 = icmp slt i64 %3, 0
  %5 = add i64 %3, 48
  %6 = select i1 %4, i64 %5, i64 %3
  %7 = icmp slt i64 %6, 0
  %8 = sub i64 -1, %6
  %9 = select i1 %7, i64 %8, i64 %6
  %10 = sdiv i64 %9, 8
  %11 = sub i64 -1, %10
  %12 = select i1 %7, i64 %11, i64 %10
  %13 = mul i64 %12, 9216
  %14 = add i64 %2, %13
  %15 = icmp slt i64 %0, 0
  %16 = sub i64 -1, %0
  %17 = select i1 %15, i64 %16, i64 %0
  %18 = sdiv i64 %17, 8
  %19 = sub i64 -1, %18
  %20 = select i1 %15, i64 %19, i64 %18
  %21 = mul i64 %20, -9216
  %22 = add i64 %14, %21
  ret i64 %22
}

declare ptr @upmemrt_dpu_alloc(i32, i32, ptr)

define ptr @forward(i64 %0, i64 %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, ptr %12, ptr %13, ptr %14, ptr %15) {
  %17 = mul i64 %0, 288
  %18 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %19 = ptrtoint ptr %18 to i64
  %20 = add i64 %19, 63
  %21 = urem i64 %20, 64
  %22 = sub i64 %20, %21
  %23 = inttoptr i64 %22 to ptr
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %18, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %23, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 0, 2
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 288, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 1, 4, 0
  %29 = getelementptr float, ptr %4, i64 %17
  call void @llvm.memcpy.p0.p0.i64(ptr %23, ptr %29, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %30 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %31 = ptrtoint ptr %30 to i64
  %32 = add i64 %31, 63
  %33 = urem i64 %32, 64
  %34 = sub i64 %32, %33
  %35 = inttoptr i64 %34 to ptr
  %36 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %37 = ptrtoint ptr %36 to i64
  %38 = add i64 %37, 63
  %39 = urem i64 %38, 64
  %40 = sub i64 %38, %39
  %41 = inttoptr i64 %40 to ptr
  %42 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %43 = ptrtoint ptr %42 to i64
  %44 = add i64 %43, 63
  %45 = urem i64 %44, 64
  %46 = sub i64 %44, %45
  %47 = inttoptr i64 %46 to ptr
  %48 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %49 = ptrtoint ptr %48 to i64
  %50 = add i64 %49, 63
  %51 = urem i64 %50, 64
  %52 = sub i64 %50, %51
  %53 = inttoptr i64 %52 to ptr
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %48, 0
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, ptr %53, 1
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 0, 2
  %57 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, i64 288, 3, 0
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %57, i64 1, 3, 1
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, i64 1, 4, 0
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 1, 4, 1
  %61 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %62 = ptrtoint ptr %61 to i64
  %63 = add i64 %62, 63
  %64 = urem i64 %63, 64
  %65 = sub i64 %63, %64
  %66 = inttoptr i64 %65 to ptr
  %67 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %68 = ptrtoint ptr %67 to i64
  %69 = add i64 %68, 63
  %70 = urem i64 %69, 64
  %71 = sub i64 %69, %70
  %72 = inttoptr i64 %71 to ptr
  %73 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %74 = ptrtoint ptr %73 to i64
  %75 = add i64 %74, 63
  %76 = urem i64 %75, 64
  %77 = sub i64 %75, %76
  %78 = inttoptr i64 %77 to ptr
  %79 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %80 = ptrtoint ptr %79 to i64
  %81 = add i64 %80, 63
  %82 = urem i64 %81, 64
  %83 = sub i64 %81, %82
  %84 = inttoptr i64 %83 to ptr
  %85 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %79, 0
  %86 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %85, ptr %84, 1
  %87 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %86, i64 0, 2
  %88 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %87, i64 288, 3, 0
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %88, i64 1, 3, 1
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, i64 1, 4, 0
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 1, 4, 1
  %92 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %93 = ptrtoint ptr %92 to i64
  %94 = add i64 %93, 63
  %95 = urem i64 %94, 64
  %96 = sub i64 %94, %95
  %97 = inttoptr i64 %96 to ptr
  %98 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %99 = ptrtoint ptr %98 to i64
  %100 = add i64 %99, 63
  %101 = urem i64 %100, 64
  %102 = sub i64 %100, %101
  %103 = inttoptr i64 %102 to ptr
  %104 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %105 = ptrtoint ptr %104 to i64
  %106 = add i64 %105, 63
  %107 = urem i64 %106, 64
  %108 = sub i64 %106, %107
  %109 = inttoptr i64 %108 to ptr
  %110 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %111 = ptrtoint ptr %110 to i64
  %112 = add i64 %111, 63
  %113 = urem i64 %112, 64
  %114 = sub i64 %112, %113
  %115 = inttoptr i64 %114 to ptr
  %116 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %110, 0
  %117 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %116, ptr %115, 1
  %118 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %117, i64 0, 2
  %119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %118, i64 288, 3, 0
  %120 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %119, i64 1, 3, 1
  %121 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %120, i64 1, 4, 0
  %122 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %121, i64 1, 4, 1
  %123 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %124 = ptrtoint ptr %123 to i64
  %125 = add i64 %124, 63
  %126 = urem i64 %125, 64
  %127 = sub i64 %125, %126
  %128 = inttoptr i64 %127 to ptr
  %129 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %130 = ptrtoint ptr %129 to i64
  %131 = add i64 %130, 63
  %132 = urem i64 %131, 64
  %133 = sub i64 %131, %132
  %134 = inttoptr i64 %133 to ptr
  %135 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %136 = ptrtoint ptr %135 to i64
  %137 = add i64 %136, 63
  %138 = urem i64 %137, 64
  %139 = sub i64 %137, %138
  %140 = inttoptr i64 %139 to ptr
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %135, 0
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, ptr %140, 1
  %143 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %142, i64 0, 2
  %144 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %143, i64 288, 3, 0
  %145 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %144, i64 1, 4, 0
  %146 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %147 = ptrtoint ptr %146 to i64
  %148 = add i64 %147, 63
  %149 = urem i64 %148, 64
  %150 = sub i64 %148, %149
  %151 = inttoptr i64 %150 to ptr
  %152 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %146, 0
  %153 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %152, ptr %151, 1
  %154 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %153, i64 0, 2
  %155 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %154, i64 288, 3, 0
  %156 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %155, i64 1, 4, 0
  %157 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %158 = ptrtoint ptr %157 to i64
  %159 = add i64 %158, 63
  %160 = urem i64 %159, 64
  %161 = sub i64 %159, %160
  %162 = inttoptr i64 %161 to ptr
  %163 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 73728) to i64), i64 64))
  %164 = ptrtoint ptr %163 to i64
  %165 = add i64 %164, 63
  %166 = urem i64 %165, 64
  %167 = sub i64 %165, %166
  %168 = inttoptr i64 %167 to ptr
  %169 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 73728) to i64), i64 64))
  %170 = ptrtoint ptr %169 to i64
  %171 = add i64 %170, 63
  %172 = urem i64 %171, 64
  %173 = sub i64 %171, %172
  %174 = inttoptr i64 %173 to ptr
  %175 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %176 = ptrtoint ptr %175 to i64
  %177 = add i64 %176, 63
  %178 = urem i64 %177, 64
  %179 = sub i64 %177, %178
  %180 = inttoptr i64 %179 to ptr
  %181 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %182 = ptrtoint ptr %181 to i64
  %183 = add i64 %182, 63
  %184 = urem i64 %183, 64
  %185 = sub i64 %183, %184
  %186 = inttoptr i64 %185 to ptr
  %187 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %181, 0
  %188 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %187, ptr %186, 1
  %189 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %188, i64 0, 2
  %190 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %189, i64 288, 3, 0
  %191 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %190, i64 1, 3, 1
  %192 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %191, i64 1, 4, 0
  %193 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %192, i64 1, 4, 1
  %194 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %195 = ptrtoint ptr %194 to i64
  %196 = add i64 %195, 63
  %197 = urem i64 %196, 64
  %198 = sub i64 %196, %197
  %199 = inttoptr i64 %198 to ptr
  %200 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %201 = ptrtoint ptr %200 to i64
  %202 = add i64 %201, 63
  %203 = urem i64 %202, 64
  %204 = sub i64 %202, %203
  %205 = inttoptr i64 %204 to ptr
  %206 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %207 = ptrtoint ptr %206 to i64
  %208 = add i64 %207, 63
  %209 = urem i64 %208, 64
  %210 = sub i64 %208, %209
  %211 = inttoptr i64 %210 to ptr
  %212 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %213 = ptrtoint ptr %212 to i64
  %214 = add i64 %213, 63
  %215 = urem i64 %214, 64
  %216 = sub i64 %214, %215
  %217 = inttoptr i64 %216 to ptr
  %218 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %219 = ptrtoint ptr %218 to i64
  %220 = add i64 %219, 63
  %221 = urem i64 %220, 64
  %222 = sub i64 %220, %221
  %223 = inttoptr i64 %222 to ptr
  %224 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %225 = ptrtoint ptr %224 to i64
  %226 = add i64 %225, 63
  %227 = urem i64 %226, 64
  %228 = sub i64 %226, %227
  %229 = inttoptr i64 %228 to ptr
  %230 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %231 = ptrtoint ptr %230 to i64
  %232 = add i64 %231, 63
  %233 = urem i64 %232, 64
  %234 = sub i64 %232, %233
  %235 = inttoptr i64 %234 to ptr
  %236 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %230, 0
  %237 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %236, ptr %235, 1
  %238 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %237, i64 0, 2
  %239 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %238, i64 768, 3, 0
  %240 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %239, i64 1, 3, 1
  %241 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %240, i64 1, 4, 0
  %242 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %241, i64 1, 4, 1
  %243 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %244 = ptrtoint ptr %243 to i64
  %245 = add i64 %244, 63
  %246 = urem i64 %245, 64
  %247 = sub i64 %245, %246
  %248 = inttoptr i64 %247 to ptr
  %249 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %250 = ptrtoint ptr %249 to i64
  %251 = add i64 %250, 63
  %252 = urem i64 %251, 64
  %253 = sub i64 %251, %252
  %254 = inttoptr i64 %253 to ptr
  %255 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %256 = ptrtoint ptr %255 to i64
  %257 = add i64 %256, 63
  %258 = urem i64 %257, 64
  %259 = sub i64 %257, %258
  %260 = inttoptr i64 %259 to ptr
  %261 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %262 = ptrtoint ptr %261 to i64
  %263 = add i64 %262, 63
  %264 = urem i64 %263, 64
  %265 = sub i64 %263, %264
  %266 = inttoptr i64 %265 to ptr
  %267 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %261, 0
  %268 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %267, ptr %266, 1
  %269 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %268, i64 0, 2
  %270 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %269, i64 768, 3, 0
  %271 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %270, i64 1, 3, 1
  %272 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %271, i64 1, 4, 0
  %273 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %272, i64 1, 4, 1
  %274 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %275 = ptrtoint ptr %274 to i64
  %276 = add i64 %275, 63
  %277 = urem i64 %276, 64
  %278 = sub i64 %276, %277
  %279 = inttoptr i64 %278 to ptr
  %280 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %281 = ptrtoint ptr %280 to i64
  %282 = add i64 %281, 63
  %283 = urem i64 %282, 64
  %284 = sub i64 %282, %283
  %285 = inttoptr i64 %284 to ptr
  %286 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %287 = ptrtoint ptr %286 to i64
  %288 = add i64 %287, 63
  %289 = urem i64 %288, 64
  %290 = sub i64 %288, %289
  %291 = inttoptr i64 %290 to ptr
  %292 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %286, 0
  %293 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %292, ptr %291, 1
  %294 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %293, i64 0, 2
  %295 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %294, i64 768, 3, 0
  %296 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %295, i64 1, 4, 0
  %297 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %298 = ptrtoint ptr %297 to i64
  %299 = add i64 %298, 63
  %300 = urem i64 %299, 64
  %301 = sub i64 %299, %300
  %302 = inttoptr i64 %301 to ptr
  %303 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %304 = ptrtoint ptr %303 to i64
  %305 = add i64 %304, 63
  %306 = urem i64 %305, 64
  %307 = sub i64 %305, %306
  %308 = inttoptr i64 %307 to ptr
  %309 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %303, 0
  %310 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %309, ptr %308, 1
  %311 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %310, i64 0, 2
  %312 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %311, i64 288, 3, 0
  %313 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %312, i64 1, 3, 1
  %314 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %313, i64 1, 4, 0
  %315 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %314, i64 1, 4, 1
  %316 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %317 = ptrtoint ptr %316 to i64
  %318 = add i64 %317, 63
  %319 = urem i64 %318, 64
  %320 = sub i64 %318, %319
  %321 = inttoptr i64 %320 to ptr
  %322 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %323 = ptrtoint ptr %322 to i64
  %324 = add i64 %323, 63
  %325 = urem i64 %324, 64
  %326 = sub i64 %324, %325
  %327 = inttoptr i64 %326 to ptr
  br label %328

328:                                              ; preds = %778, %16
  %329 = phi i64 [ %792, %778 ], [ 0, %16 ]
  %330 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %791, %778 ], [ %28, %16 ]
  %331 = icmp slt i64 %329, 6
  br i1 %331, label %332, label %793

332:                                              ; preds = %328
  %333 = mul i64 %329, 288
  %334 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, 3, 0
  %335 = mul i64 %334, 1
  %336 = mul i64 %335, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %337 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, 1
  %338 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, 2
  %339 = getelementptr float, ptr %337, i64 %338
  call void @llvm.memcpy.p0.p0.i64(ptr %35, ptr %339, i64 %336, i1 false)
  %340 = getelementptr float, ptr %5, i64 %333
  call void @llvm.memcpy.p0.p0.i64(ptr %41, ptr %340, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %341 = call ptr @rmsnorm(ptr %35, ptr %41)
  call void @llvm.memcpy.p0.p0.i64(ptr %53, ptr %47, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %342

342:                                              ; preds = %361, %332
  %343 = phi i64 [ %385, %361 ], [ 0, %332 ]
  %344 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %375, %361 ], [ %60, %332 ]
  %345 = icmp slt i64 %343, 288
  br i1 %345, label %346, label %386

346:                                              ; preds = %342
  %347 = mul i64 %329, 82944
  %348 = mul i64 %343, 288
  %349 = add i64 %347, %348
  %350 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program)
  br label %351

351:                                              ; preds = %354, %346
  %352 = phi i64 [ %360, %354 ], [ 0, %346 ]
  %353 = icmp slt i64 %352, 288
  br i1 %353, label %354, label %361

354:                                              ; preds = %351
  %355 = add i64 %352, 0
  %356 = getelementptr float, ptr %341, i64 %355
  %357 = load float, ptr %356, align 4
  %358 = add i64 0, %352
  %359 = getelementptr float, ptr %66, i64 %358
  store float %357, ptr %359, align 4
  %360 = add i64 %352, 1
  br label %351

361:                                              ; preds = %351
  %362 = getelementptr inbounds float, ptr %6, i64 %349
  call void @upmemrt_dpu_scatter(ptr %350, ptr %362, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %350, ptr %66, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %350, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %350)
  call void @upmemrt_dpu_gather(ptr %350, ptr %72, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %350)
  %363 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %364 = ptrtoint ptr %363 to i64
  %365 = add i64 %364, 63
  %366 = urem i64 %365, 64
  %367 = sub i64 %365, %366
  %368 = inttoptr i64 %367 to ptr
  %369 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %363, 0
  %370 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %369, ptr %368, 1
  %371 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %370, i64 0, 2
  %372 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %371, i64 288, 3, 0
  %373 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %372, i64 1, 3, 1
  %374 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %373, i64 1, 4, 0
  %375 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %374, i64 1, 4, 1
  %376 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, 3, 0
  %377 = mul i64 %376, 1
  %378 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, 3, 1
  %379 = mul i64 %377, %378
  %380 = mul i64 %379, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %381 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, 1
  %382 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, 2
  %383 = getelementptr float, ptr %381, i64 %382
  call void @llvm.memcpy.p0.p0.i64(ptr %368, ptr %383, i64 %380, i1 false)
  %384 = getelementptr float, ptr %368, i64 %343
  call void @llvm.memcpy.p0.p0.i64(ptr %384, ptr %72, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %385 = add i64 %343, 48
  br label %342

386:                                              ; preds = %342
  %387 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %344, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %84, ptr %78, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %388

388:                                              ; preds = %407, %386
  %389 = phi i64 [ %431, %407 ], [ 0, %386 ]
  %390 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %421, %407 ], [ %91, %386 ]
  %391 = icmp slt i64 %389, 288
  br i1 %391, label %392, label %432

392:                                              ; preds = %388
  %393 = mul i64 %329, 82944
  %394 = mul i64 %389, 288
  %395 = add i64 %393, %394
  %396 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_0)
  br label %397

397:                                              ; preds = %400, %392
  %398 = phi i64 [ %406, %400 ], [ 0, %392 ]
  %399 = icmp slt i64 %398, 288
  br i1 %399, label %400, label %407

400:                                              ; preds = %397
  %401 = add i64 %398, 0
  %402 = getelementptr float, ptr %341, i64 %401
  %403 = load float, ptr %402, align 4
  %404 = add i64 0, %398
  %405 = getelementptr float, ptr %97, i64 %404
  store float %403, ptr %405, align 4
  %406 = add i64 %398, 1
  br label %397

407:                                              ; preds = %397
  %408 = getelementptr inbounds float, ptr %7, i64 %395
  call void @upmemrt_dpu_scatter(ptr %396, ptr %408, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %396, ptr %97, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %396, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %396)
  call void @upmemrt_dpu_gather(ptr %396, ptr %103, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %396)
  %409 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %410 = ptrtoint ptr %409 to i64
  %411 = add i64 %410, 63
  %412 = urem i64 %411, 64
  %413 = sub i64 %411, %412
  %414 = inttoptr i64 %413 to ptr
  %415 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %409, 0
  %416 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %415, ptr %414, 1
  %417 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %416, i64 0, 2
  %418 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %417, i64 288, 3, 0
  %419 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %418, i64 1, 3, 1
  %420 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %419, i64 1, 4, 0
  %421 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %420, i64 1, 4, 1
  %422 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, 3, 0
  %423 = mul i64 %422, 1
  %424 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, 3, 1
  %425 = mul i64 %423, %424
  %426 = mul i64 %425, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %427 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, 1
  %428 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, 2
  %429 = getelementptr float, ptr %427, i64 %428
  call void @llvm.memcpy.p0.p0.i64(ptr %414, ptr %429, i64 %426, i1 false)
  %430 = getelementptr float, ptr %414, i64 %389
  call void @llvm.memcpy.p0.p0.i64(ptr %430, ptr %103, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %431 = add i64 %389, 48
  br label %388

432:                                              ; preds = %388
  %433 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %390, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %115, ptr %109, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %434

434:                                              ; preds = %453, %432
  %435 = phi i64 [ %477, %453 ], [ 0, %432 ]
  %436 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %467, %453 ], [ %122, %432 ]
  %437 = icmp slt i64 %435, 288
  br i1 %437, label %438, label %478

438:                                              ; preds = %434
  %439 = mul i64 %329, 82944
  %440 = mul i64 %435, 288
  %441 = add i64 %439, %440
  %442 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_1)
  br label %443

443:                                              ; preds = %446, %438
  %444 = phi i64 [ %452, %446 ], [ 0, %438 ]
  %445 = icmp slt i64 %444, 288
  br i1 %445, label %446, label %453

446:                                              ; preds = %443
  %447 = add i64 %444, 0
  %448 = getelementptr float, ptr %341, i64 %447
  %449 = load float, ptr %448, align 4
  %450 = add i64 0, %444
  %451 = getelementptr float, ptr %128, i64 %450
  store float %449, ptr %451, align 4
  %452 = add i64 %444, 1
  br label %443

453:                                              ; preds = %443
  %454 = getelementptr inbounds float, ptr %8, i64 %441
  call void @upmemrt_dpu_scatter(ptr %442, ptr %454, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %442, ptr %128, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %442, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %442)
  call void @upmemrt_dpu_gather(ptr %442, ptr %134, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %442)
  %455 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %456 = ptrtoint ptr %455 to i64
  %457 = add i64 %456, 63
  %458 = urem i64 %457, 64
  %459 = sub i64 %457, %458
  %460 = inttoptr i64 %459 to ptr
  %461 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %455, 0
  %462 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %461, ptr %460, 1
  %463 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %462, i64 0, 2
  %464 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %463, i64 288, 3, 0
  %465 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %464, i64 1, 3, 1
  %466 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %465, i64 1, 4, 0
  %467 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %466, i64 1, 4, 1
  %468 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 3, 0
  %469 = mul i64 %468, 1
  %470 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 3, 1
  %471 = mul i64 %469, %470
  %472 = mul i64 %471, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %473 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 1
  %474 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 2
  %475 = getelementptr float, ptr %473, i64 %474
  call void @llvm.memcpy.p0.p0.i64(ptr %460, ptr %475, i64 %472, i1 false)
  %476 = getelementptr float, ptr %460, i64 %435
  call void @llvm.memcpy.p0.p0.i64(ptr %476, ptr %134, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %477 = add i64 %435, 48
  br label %434

478:                                              ; preds = %434
  %479 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %436, 1
  %480 = uitofp i64 %1 to float
  call void @llvm.memcpy.p0.p0.i64(ptr %140, ptr %387, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %151, ptr %433, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %481

481:                                              ; preds = %536, %478
  %482 = phi i64 [ %537, %536 ], [ 0, %478 ]
  %483 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %512, %536 ], [ %145, %478 ]
  %484 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %535, %536 ], [ %156, %478 ]
  %485 = icmp slt i64 %482, 288
  br i1 %485, label %486, label %538

486:                                              ; preds = %481
  %487 = urem i64 %482, 48
  %488 = uitofp i64 %487 to float
  %489 = fdiv float %488, 4.800000e+01
  %490 = call float @llvm.pow.f32(float 1.000000e+04, float %489)
  %491 = fdiv float 1.000000e+00, %490
  %492 = fmul float %480, %491
  %493 = call float @llvm.cos.f32(float %492)
  %494 = call float @llvm.sin.f32(float %492)
  %495 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %496 = ptrtoint ptr %495 to i64
  %497 = add i64 %496, 63
  %498 = urem i64 %497, 64
  %499 = sub i64 %497, %498
  %500 = inttoptr i64 %499 to ptr
  %501 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %483, 3, 0
  %502 = mul i64 %501, 1
  %503 = mul i64 %502, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %504 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %483, 1
  %505 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %483, 2
  %506 = getelementptr float, ptr %504, i64 %505
  call void @llvm.memcpy.p0.p0.i64(ptr %500, ptr %506, i64 %503, i1 false)
  %507 = call ptr @rot(ptr %500, i64 %482, float %493, float %494)
  %508 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %507, 0
  %509 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %508, ptr %507, 1
  %510 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %509, i64 0, 2
  %511 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %510, i64 288, 3, 0
  %512 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %511, i64 1, 4, 0
  %513 = icmp ult i64 %482, 288
  %514 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %515 = ptrtoint ptr %514 to i64
  %516 = add i64 %515, 63
  %517 = urem i64 %516, 64
  %518 = sub i64 %516, %517
  %519 = inttoptr i64 %518 to ptr
  br i1 %513, label %520, label %533

520:                                              ; preds = %486
  %521 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, 3, 0
  %522 = mul i64 %521, 1
  %523 = mul i64 %522, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %524 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, 1
  %525 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, 2
  %526 = getelementptr float, ptr %524, i64 %525
  call void @llvm.memcpy.p0.p0.i64(ptr %519, ptr %526, i64 %523, i1 false)
  %527 = call ptr @rot(ptr %519, i64 %482, float %493, float %494)
  %528 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %527, 0
  %529 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %528, ptr %527, 1
  %530 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %529, i64 0, 2
  %531 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %530, i64 288, 3, 0
  %532 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %531, i64 1, 4, 0
  br label %534

533:                                              ; preds = %486
  br label %534

534:                                              ; preds = %520, %533
  %535 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %484, %533 ], [ %532, %520 ]
  br label %536

536:                                              ; preds = %534
  %537 = add i64 %482, 2
  br label %481

538:                                              ; preds = %481
  %539 = mul i64 %329, 73728
  %540 = mul i64 %1, 288
  %541 = add i64 %539, %540
  %542 = getelementptr float, ptr %3, i64 %541
  call void @llvm.memcpy.p0.p0.i64(ptr %542, ptr %479, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %543 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, 3, 0
  %544 = mul i64 %543, 1
  %545 = mul i64 %544, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %546 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, 1
  %547 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %484, 2
  %548 = getelementptr float, ptr %546, i64 %547
  %549 = getelementptr float, ptr %2, i64 %541
  call void @llvm.memcpy.p0.p0.i64(ptr %549, ptr %548, i64 %545, i1 false)
  %550 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %483, 3, 0
  %551 = mul i64 %550, 1
  %552 = mul i64 %551, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %553 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %483, 1
  %554 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %483, 2
  %555 = getelementptr float, ptr %553, i64 %554
  call void @llvm.memcpy.p0.p0.i64(ptr %162, ptr %555, i64 %552, i1 false)
  %556 = getelementptr float, ptr %2, i64 %539
  call void @llvm.memcpy.p0.p0.i64(ptr %168, ptr %556, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 73728), i1 false)
  %557 = getelementptr float, ptr %3, i64 %539
  call void @llvm.memcpy.p0.p0.i64(ptr %174, ptr %557, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 73728), i1 false)
  %558 = call ptr @mha(ptr %162, ptr %168, ptr %174, i64 %1)
  call void @llvm.memcpy.p0.p0.i64(ptr %186, ptr %180, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %559

559:                                              ; preds = %578, %538
  %560 = phi i64 [ %602, %578 ], [ 0, %538 ]
  %561 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %592, %578 ], [ %193, %538 ]
  %562 = icmp slt i64 %560, 288
  br i1 %562, label %563, label %603

563:                                              ; preds = %559
  %564 = mul i64 %329, 82944
  %565 = mul i64 %560, 288
  %566 = add i64 %564, %565
  %567 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_2)
  br label %568

568:                                              ; preds = %571, %563
  %569 = phi i64 [ %577, %571 ], [ 0, %563 ]
  %570 = icmp slt i64 %569, 288
  br i1 %570, label %571, label %578

571:                                              ; preds = %568
  %572 = add i64 %569, 0
  %573 = getelementptr float, ptr %558, i64 %572
  %574 = load float, ptr %573, align 4
  %575 = add i64 0, %569
  %576 = getelementptr float, ptr %199, i64 %575
  store float %574, ptr %576, align 4
  %577 = add i64 %569, 1
  br label %568

578:                                              ; preds = %568
  %579 = getelementptr inbounds float, ptr %9, i64 %566
  call void @upmemrt_dpu_scatter(ptr %567, ptr %579, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %567, ptr %199, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %567, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %567)
  call void @upmemrt_dpu_gather(ptr %567, ptr %205, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %567)
  %580 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %581 = ptrtoint ptr %580 to i64
  %582 = add i64 %581, 63
  %583 = urem i64 %582, 64
  %584 = sub i64 %582, %583
  %585 = inttoptr i64 %584 to ptr
  %586 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %580, 0
  %587 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %586, ptr %585, 1
  %588 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %587, i64 0, 2
  %589 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %588, i64 288, 3, 0
  %590 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %589, i64 1, 3, 1
  %591 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %590, i64 1, 4, 0
  %592 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %591, i64 1, 4, 1
  %593 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %561, 3, 0
  %594 = mul i64 %593, 1
  %595 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %561, 3, 1
  %596 = mul i64 %594, %595
  %597 = mul i64 %596, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %598 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %561, 1
  %599 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %561, 2
  %600 = getelementptr float, ptr %598, i64 %599
  call void @llvm.memcpy.p0.p0.i64(ptr %585, ptr %600, i64 %597, i1 false)
  %601 = getelementptr float, ptr %585, i64 %560
  call void @llvm.memcpy.p0.p0.i64(ptr %601, ptr %205, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %602 = add i64 %560, 48
  br label %559

603:                                              ; preds = %559
  %604 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %561, 1
  %605 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_3)
  call void @upmemrt_dpu_scatter(ptr %605, ptr %337, i64 4, i64 288, i64 6, i64 192, i64 0, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %605, ptr %604, i64 4, i64 288, i64 6, i64 192, i64 192, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %605, ptr @__constant_48x6xf32, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  call void @upmemrt_dpu_launch(ptr %605)
  call void @upmemrt_dpu_gather(ptr %605, ptr %211, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  call void @upmemrt_dpu_free(ptr %605)
  call void @llvm.memcpy.p0.p0.i64(ptr %217, ptr %211, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %606 = getelementptr float, ptr %13, i64 %333
  call void @llvm.memcpy.p0.p0.i64(ptr %223, ptr %606, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %607 = call ptr @rmsnorm(ptr %217, ptr %223)
  call void @llvm.memcpy.p0.p0.i64(ptr %235, ptr %229, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %608

608:                                              ; preds = %627, %603
  %609 = phi i64 [ %651, %627 ], [ 0, %603 ]
  %610 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %641, %627 ], [ %242, %603 ]
  %611 = icmp slt i64 %609, 768
  br i1 %611, label %612, label %652

612:                                              ; preds = %608
  %613 = mul i64 %329, 221184
  %614 = mul i64 %609, 288
  %615 = add i64 %613, %614
  %616 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_4)
  br label %617

617:                                              ; preds = %620, %612
  %618 = phi i64 [ %626, %620 ], [ 0, %612 ]
  %619 = icmp slt i64 %618, 288
  br i1 %619, label %620, label %627

620:                                              ; preds = %617
  %621 = add i64 %618, 0
  %622 = getelementptr float, ptr %607, i64 %621
  %623 = load float, ptr %622, align 4
  %624 = add i64 0, %618
  %625 = getelementptr float, ptr %248, i64 %624
  store float %623, ptr %625, align 4
  %626 = add i64 %618, 1
  br label %617

627:                                              ; preds = %617
  %628 = getelementptr inbounds float, ptr %10, i64 %615
  call void @upmemrt_dpu_scatter(ptr %616, ptr %628, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %616, ptr %248, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %616, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %616)
  call void @upmemrt_dpu_gather(ptr %616, ptr %254, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %616)
  %629 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %630 = ptrtoint ptr %629 to i64
  %631 = add i64 %630, 63
  %632 = urem i64 %631, 64
  %633 = sub i64 %631, %632
  %634 = inttoptr i64 %633 to ptr
  %635 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %629, 0
  %636 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %635, ptr %634, 1
  %637 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %636, i64 0, 2
  %638 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %637, i64 768, 3, 0
  %639 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %638, i64 1, 3, 1
  %640 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %639, i64 1, 4, 0
  %641 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %640, i64 1, 4, 1
  %642 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %610, 3, 0
  %643 = mul i64 %642, 1
  %644 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %610, 3, 1
  %645 = mul i64 %643, %644
  %646 = mul i64 %645, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %647 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %610, 1
  %648 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %610, 2
  %649 = getelementptr float, ptr %647, i64 %648
  call void @llvm.memcpy.p0.p0.i64(ptr %634, ptr %649, i64 %646, i1 false)
  %650 = getelementptr float, ptr %634, i64 %609
  call void @llvm.memcpy.p0.p0.i64(ptr %650, ptr %254, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %651 = add i64 %609, 48
  br label %608

652:                                              ; preds = %608
  %653 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %610, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %266, ptr %260, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %654

654:                                              ; preds = %673, %652
  %655 = phi i64 [ %697, %673 ], [ 0, %652 ]
  %656 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %687, %673 ], [ %273, %652 ]
  %657 = icmp slt i64 %655, 768
  br i1 %657, label %658, label %698

658:                                              ; preds = %654
  %659 = mul i64 %329, 221184
  %660 = mul i64 %655, 288
  %661 = add i64 %659, %660
  %662 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_5)
  br label %663

663:                                              ; preds = %666, %658
  %664 = phi i64 [ %672, %666 ], [ 0, %658 ]
  %665 = icmp slt i64 %664, 288
  br i1 %665, label %666, label %673

666:                                              ; preds = %663
  %667 = add i64 %664, 0
  %668 = getelementptr float, ptr %607, i64 %667
  %669 = load float, ptr %668, align 4
  %670 = add i64 0, %664
  %671 = getelementptr float, ptr %279, i64 %670
  store float %669, ptr %671, align 4
  %672 = add i64 %664, 1
  br label %663

673:                                              ; preds = %663
  %674 = getelementptr inbounds float, ptr %12, i64 %661
  call void @upmemrt_dpu_scatter(ptr %662, ptr %674, i64 4, i64 13824, i64 288, i64 9216, i64 0, ptr @scatter_map)
  call void @upmemrt_dpu_scatter(ptr %662, ptr %279, i64 4, i64 288, i64 6, i64 9216, i64 9216, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %662, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %662)
  call void @upmemrt_dpu_gather(ptr %662, ptr %285, i64 4, i64 48, i64 1, i64 32, i64 18432, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %662)
  %675 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %676 = ptrtoint ptr %675 to i64
  %677 = add i64 %676, 63
  %678 = urem i64 %677, 64
  %679 = sub i64 %677, %678
  %680 = inttoptr i64 %679 to ptr
  %681 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %675, 0
  %682 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %681, ptr %680, 1
  %683 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %682, i64 0, 2
  %684 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %683, i64 768, 3, 0
  %685 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %684, i64 1, 3, 1
  %686 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %685, i64 1, 4, 0
  %687 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %686, i64 1, 4, 1
  %688 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %656, 3, 0
  %689 = mul i64 %688, 1
  %690 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %656, 3, 1
  %691 = mul i64 %689, %690
  %692 = mul i64 %691, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %693 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %656, 1
  %694 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %656, 2
  %695 = getelementptr float, ptr %693, i64 %694
  call void @llvm.memcpy.p0.p0.i64(ptr %680, ptr %695, i64 %692, i1 false)
  %696 = getelementptr float, ptr %680, i64 %655
  call void @llvm.memcpy.p0.p0.i64(ptr %696, ptr %285, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %697 = add i64 %655, 48
  br label %654

698:                                              ; preds = %654
  %699 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %656, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %291, ptr %653, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 768), i1 false)
  br label %700

700:                                              ; preds = %704, %698
  %701 = phi i64 [ %731, %704 ], [ 0, %698 ]
  %702 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %724, %704 ], [ %296, %698 ]
  %703 = icmp slt i64 %701, 768
  br i1 %703, label %704, label %732

704:                                              ; preds = %700
  %705 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %702, 1
  %706 = getelementptr float, ptr %705, i64 %701
  %707 = load float, ptr %706, align 4
  %708 = getelementptr float, ptr %699, i64 %701
  %709 = load float, ptr %708, align 4
  %710 = call float @llvm.exp.f32(float %707)
  %711 = fadd float %710, 1.000000e+00
  %712 = fdiv float 1.000000e+00, %711
  %713 = fmul float %709, %712
  %714 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 768) to i64), i64 64))
  %715 = ptrtoint ptr %714 to i64
  %716 = add i64 %715, 63
  %717 = urem i64 %716, 64
  %718 = sub i64 %716, %717
  %719 = inttoptr i64 %718 to ptr
  %720 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %714, 0
  %721 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %720, ptr %719, 1
  %722 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %721, i64 0, 2
  %723 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %722, i64 768, 3, 0
  %724 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %723, i64 1, 4, 0
  %725 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %702, 3, 0
  %726 = mul i64 %725, 1
  %727 = mul i64 %726, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %728 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %702, 2
  %729 = getelementptr float, ptr %705, i64 %728
  call void @llvm.memcpy.p0.p0.i64(ptr %719, ptr %729, i64 %727, i1 false)
  %730 = getelementptr float, ptr %719, i64 %701
  store float %713, ptr %730, align 4
  %731 = add i64 %701, 1
  br label %700

732:                                              ; preds = %700
  %733 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %702, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %308, ptr %302, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  br label %734

734:                                              ; preds = %753, %732
  %735 = phi i64 [ %777, %753 ], [ 0, %732 ]
  %736 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %767, %753 ], [ %315, %732 ]
  %737 = icmp slt i64 %735, 288
  br i1 %737, label %738, label %778

738:                                              ; preds = %734
  %739 = mul i64 %329, 221184
  %740 = mul i64 %735, 768
  %741 = add i64 %739, %740
  %742 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_6)
  br label %743

743:                                              ; preds = %746, %738
  %744 = phi i64 [ %752, %746 ], [ 0, %738 ]
  %745 = icmp slt i64 %744, 768
  br i1 %745, label %746, label %753

746:                                              ; preds = %743
  %747 = add i64 %744, 0
  %748 = getelementptr float, ptr %733, i64 %747
  %749 = load float, ptr %748, align 4
  %750 = add i64 0, %744
  %751 = getelementptr float, ptr %321, i64 %750
  store float %749, ptr %751, align 4
  %752 = add i64 %744, 1
  br label %743

753:                                              ; preds = %743
  %754 = getelementptr inbounds float, ptr %11, i64 %741
  call void @upmemrt_dpu_scatter(ptr %742, ptr %754, i64 4, i64 36864, i64 768, i64 24576, i64 0, ptr @scatter_map_3)
  call void @upmemrt_dpu_scatter(ptr %742, ptr %321, i64 4, i64 768, i64 16, i64 24576, i64 24576, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %742, ptr @__constant_48x1xf32, i64 4, i64 48, i64 1, i64 32, i64 49152, ptr @scatter_map_1)
  call void @upmemrt_dpu_launch(ptr %742)
  call void @upmemrt_dpu_gather(ptr %742, ptr %327, i64 4, i64 48, i64 1, i64 32, i64 49152, ptr @scatter_map_1)
  call void @upmemrt_dpu_free(ptr %742)
  %755 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %756 = ptrtoint ptr %755 to i64
  %757 = add i64 %756, 63
  %758 = urem i64 %757, 64
  %759 = sub i64 %757, %758
  %760 = inttoptr i64 %759 to ptr
  %761 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %755, 0
  %762 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %761, ptr %760, 1
  %763 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %762, i64 0, 2
  %764 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %763, i64 288, 3, 0
  %765 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %764, i64 1, 3, 1
  %766 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %765, i64 1, 4, 0
  %767 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %766, i64 1, 4, 1
  %768 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %736, 3, 0
  %769 = mul i64 %768, 1
  %770 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %736, 3, 1
  %771 = mul i64 %769, %770
  %772 = mul i64 %771, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %773 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %736, 1
  %774 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %736, 2
  %775 = getelementptr float, ptr %773, i64 %774
  call void @llvm.memcpy.p0.p0.i64(ptr %760, ptr %775, i64 %772, i1 false)
  %776 = getelementptr float, ptr %760, i64 %735
  call void @llvm.memcpy.p0.p0.i64(ptr %776, ptr %327, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %777 = add i64 %735, 48
  br label %734

778:                                              ; preds = %734
  %779 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %736, 1
  %780 = call ptr @upmemrt_dpu_alloc(i32 1, i32 6, ptr @dpu_program_7)
  call void @upmemrt_dpu_scatter(ptr %780, ptr %337, i64 4, i64 288, i64 6, i64 192, i64 0, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %780, ptr %779, i64 4, i64 288, i64 6, i64 192, i64 192, ptr @scatter_map_2)
  call void @upmemrt_dpu_scatter(ptr %780, ptr @__constant_48x6xf32, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  call void @upmemrt_dpu_launch(ptr %780)
  %781 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %782 = ptrtoint ptr %781 to i64
  %783 = add i64 %782, 63
  %784 = urem i64 %783, 64
  %785 = sub i64 %783, %784
  %786 = inttoptr i64 %785 to ptr
  call void @upmemrt_dpu_gather(ptr %780, ptr %786, i64 4, i64 288, i64 6, i64 192, i64 384, ptr @scatter_map_2)
  %787 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %781, 0
  %788 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %787, ptr %786, 1
  %789 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %788, i64 0, 2
  %790 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %789, i64 288, 3, 0
  %791 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %790, i64 1, 4, 0
  call void @upmemrt_dpu_free(ptr %780)
  %792 = add i64 %329, 1
  br label %328

793:                                              ; preds = %328
  %794 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %795 = ptrtoint ptr %794 to i64
  %796 = add i64 %795, 63
  %797 = urem i64 %796, 64
  %798 = sub i64 %796, %797
  %799 = inttoptr i64 %798 to ptr
  %800 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, 3, 0
  %801 = mul i64 %800, 1
  %802 = mul i64 %801, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %803 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, 1
  %804 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %330, 2
  %805 = getelementptr float, ptr %803, i64 %804
  call void @llvm.memcpy.p0.p0.i64(ptr %799, ptr %805, i64 %802, i1 false)
  %806 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %807 = ptrtoint ptr %806 to i64
  %808 = add i64 %807, 63
  %809 = urem i64 %808, 64
  %810 = sub i64 %808, %809
  %811 = inttoptr i64 %810 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %811, ptr %14, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %812 = call ptr @rmsnorm(ptr %799, ptr %811)
  %813 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 9437184) to i64), i64 64))
  %814 = ptrtoint ptr %813 to i64
  %815 = add i64 %814, 63
  %816 = urem i64 %815, 64
  %817 = sub i64 %815, %816
  %818 = inttoptr i64 %817 to ptr
  br label %819

819:                                              ; preds = %831, %793
  %820 = phi i64 [ %832, %831 ], [ 0, %793 ]
  %821 = icmp slt i64 %820, 32768
  br i1 %821, label %822, label %833

822:                                              ; preds = %819
  br label %823

823:                                              ; preds = %826, %822
  %824 = phi i64 [ %830, %826 ], [ 0, %822 ]
  %825 = icmp slt i64 %824, 288
  br i1 %825, label %826, label %831

826:                                              ; preds = %823
  %827 = mul i64 %820, 288
  %828 = add i64 %827, %824
  %829 = getelementptr float, ptr %818, i64 %828
  store float 0.000000e+00, ptr %829, align 4
  %830 = add i64 %824, 1
  br label %823

831:                                              ; preds = %823
  %832 = add i64 %820, 1
  br label %819

833:                                              ; preds = %819
  call void @llvm.memcpy.p0.p0.i64(ptr %818, ptr %15, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 9216000), i1 false)
  %834 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64), i64 64))
  %835 = ptrtoint ptr %834 to i64
  %836 = add i64 %835, 63
  %837 = urem i64 %836, 64
  %838 = sub i64 %836, %837
  %839 = inttoptr i64 %838 to ptr
  %840 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64), i64 64))
  %841 = ptrtoint ptr %840 to i64
  %842 = add i64 %841, 63
  %843 = urem i64 %842, 64
  %844 = sub i64 %842, %843
  %845 = inttoptr i64 %844 to ptr
  %846 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %840, 0
  %847 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %846, ptr %845, 1
  %848 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %847, i64 0, 2
  %849 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %848, i64 32768, 3, 0
  %850 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %849, i64 1, 3, 1
  %851 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %850, i64 1, 4, 0
  %852 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %851, i64 1, 4, 1
  call void @llvm.memcpy.p0.p0.i64(ptr %845, ptr %839, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 32768), i1 false)
  %853 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %854 = ptrtoint ptr %853 to i64
  %855 = add i64 %854, 63
  %856 = urem i64 %855, 64
  %857 = sub i64 %855, %856
  %858 = inttoptr i64 %857 to ptr
  %859 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %860 = ptrtoint ptr %859 to i64
  %861 = add i64 %860, 63
  %862 = urem i64 %861, 64
  %863 = sub i64 %861, %862
  %864 = inttoptr i64 %863 to ptr
  br label %865

865:                                              ; preds = %882, %833
  %866 = phi i64 [ %906, %882 ], [ 0, %833 ]
  %867 = phi { ptr, ptr, i64, [2 x i64], [2 x i64] } [ %896, %882 ], [ %852, %833 ]
  %868 = icmp slt i64 %866, 32768
  br i1 %868, label %869, label %907

869:                                              ; preds = %865
  %870 = mul i64 %866, 288
  %871 = call ptr @upmemrt_dpu_alloc(i32 2, i32 8, ptr @dpu_program_8)
  br label %872

872:                                              ; preds = %875, %869
  %873 = phi i64 [ %881, %875 ], [ 0, %869 ]
  %874 = icmp slt i64 %873, 288
  br i1 %874, label %875, label %882

875:                                              ; preds = %872
  %876 = add i64 %873, 0
  %877 = getelementptr float, ptr %812, i64 %876
  %878 = load float, ptr %877, align 4
  %879 = add i64 0, %873
  %880 = getelementptr float, ptr %858, i64 %879
  store float %878, ptr %880, align 4
  %881 = add i64 %873, 1
  br label %872

882:                                              ; preds = %872
  %883 = getelementptr inbounds float, ptr %818, i64 %870
  call void @upmemrt_dpu_scatter(ptr %871, ptr %883, i64 4, i64 73728, i64 288, i64 18432, i64 0, ptr @scatter_map_4)
  call void @upmemrt_dpu_scatter(ptr %871, ptr %858, i64 4, i64 288, i64 1, i64 18432, i64 18432, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %871, ptr @__constant_256x1xf32, i64 4, i64 256, i64 1, i64 64, i64 36864, ptr @scatter_map_5)
  call void @upmemrt_dpu_launch(ptr %871)
  call void @upmemrt_dpu_gather(ptr %871, ptr %864, i64 4, i64 256, i64 1, i64 64, i64 36864, ptr @scatter_map_5)
  call void @upmemrt_dpu_free(ptr %871)
  %884 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 32768) to i64), i64 64))
  %885 = ptrtoint ptr %884 to i64
  %886 = add i64 %885, 63
  %887 = urem i64 %886, 64
  %888 = sub i64 %886, %887
  %889 = inttoptr i64 %888 to ptr
  %890 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %884, 0
  %891 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %890, ptr %889, 1
  %892 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %891, i64 0, 2
  %893 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %892, i64 32768, 3, 0
  %894 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %893, i64 1, 3, 1
  %895 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %894, i64 1, 4, 0
  %896 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %895, i64 1, 4, 1
  %897 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %867, 3, 0
  %898 = mul i64 %897, 1
  %899 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %867, 3, 1
  %900 = mul i64 %898, %899
  %901 = mul i64 %900, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %902 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %867, 1
  %903 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %867, 2
  %904 = getelementptr float, ptr %902, i64 %903
  call void @llvm.memcpy.p0.p0.i64(ptr %889, ptr %904, i64 %901, i1 false)
  %905 = getelementptr float, ptr %889, i64 %866
  call void @llvm.memcpy.p0.p0.i64(ptr %905, ptr %864, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 256), i1 false)
  %906 = add i64 %866, 256
  br label %865

907:                                              ; preds = %865
  %908 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %867, 0
  ret ptr %908
}

define ptr @rot(ptr %0, i64 %1, float %2, float %3) {
  %5 = add i64 %1, 1
  %6 = getelementptr float, ptr %0, i64 %1
  %7 = load float, ptr %6, align 4
  %8 = getelementptr float, ptr %0, i64 %5
  %9 = load float, ptr %8, align 4
  %10 = fmul float %7, %2
  %11 = fmul float %9, %3
  %12 = fsub float %10, %11
  %13 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %14 = ptrtoint ptr %13 to i64
  %15 = add i64 %14, 63
  %16 = urem i64 %15, 64
  %17 = sub i64 %15, %16
  %18 = inttoptr i64 %17 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %18, ptr %0, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %19 = getelementptr float, ptr %18, i64 %1
  store float %12, ptr %19, align 4
  %20 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %21 = ptrtoint ptr %20 to i64
  %22 = add i64 %21, 63
  %23 = urem i64 %22, 64
  %24 = sub i64 %22, %23
  %25 = inttoptr i64 %24 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %25, ptr %18, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %26 = getelementptr float, ptr %25, i64 %1
  store float %12, ptr %26, align 4
  ret ptr %20
}

define ptr @mha(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %6 = ptrtoint ptr %5 to i64
  %7 = add i64 %6, 63
  %8 = urem i64 %7, 64
  %9 = sub i64 %7, %8
  %10 = inttoptr i64 %9 to ptr
  %11 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 288, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %16, ptr %10, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 288), i1 false)
  %22 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %23 = ptrtoint ptr %22 to i64
  %24 = add i64 %23, 63
  %25 = urem i64 %24, 64
  %26 = sub i64 %24, %25
  %27 = inttoptr i64 %26 to ptr
  %28 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 12288) to i64), i64 64))
  %29 = ptrtoint ptr %28 to i64
  %30 = add i64 %29, 63
  %31 = urem i64 %30, 64
  %32 = sub i64 %30, %31
  %33 = inttoptr i64 %32 to ptr
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %28, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %33, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 0, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 256, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 48, 3, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 48, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 1, 4, 1
  %41 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 12288) to i64), i64 64))
  %42 = ptrtoint ptr %41 to i64
  %43 = add i64 %42, 63
  %44 = urem i64 %43, 64
  %45 = sub i64 %43, %44
  %46 = inttoptr i64 %45 to ptr
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %41, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, ptr %46, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 0, 2
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 256, 3, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 48, 3, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 48, 4, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 1, 4, 1
  br label %54

54:                                               ; preds = %58, %4
  %55 = phi i64 [ %108, %58 ], [ 0, %4 ]
  %56 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %100, %58 ], [ %21, %4 ]
  %57 = icmp slt i64 %55, 6
  br i1 %57, label %58, label %109

58:                                               ; preds = %54
  %59 = mul i64 %55, 48
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %1, 0
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, ptr %1, 1
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 %59, 2
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 256, 3, 0
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 288, 4, 0
  %65 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, i64 48, 3, 1
  %66 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %65, i64 1, 4, 1
  %67 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %2, 0
  %68 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %67, ptr %2, 1
  %69 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %68, i64 %59, 2
  %70 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %69, i64 256, 3, 0
  %71 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %70, i64 288, 4, 0
  %72 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %71, i64 48, 3, 1
  %73 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %72, i64 1, 4, 1
  %74 = getelementptr float, ptr %0, i64 %59
  call void @llvm.memcpy.p0.p0.i64(ptr %27, ptr %74, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %75 = call ptr @llvm.stacksave.p0()
  %76 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %66, ptr %76, align 8
  %77 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %76, 1
  %78 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %78, align 8
  %79 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %78, 1
  %80 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %77, ptr %80, align 8
  %81 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %79, ptr %81, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %80, ptr %81)
  call void @llvm.stackrestore.p0(ptr %75)
  %82 = call ptr @llvm.stacksave.p0()
  %83 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %73, ptr %83, align 8
  %84 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %83, 1
  %85 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, ptr %85, align 8
  %86 = insertvalue { i64, ptr } { i64 2, ptr undef }, ptr %85, 1
  %87 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %84, ptr %87, align 8
  %88 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %86, ptr %88, align 8
  call void @memrefCopy(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), ptr %87, ptr %88)
  call void @llvm.stackrestore.p0(ptr %82)
  %89 = call ptr @attn(ptr %27, ptr %33, ptr %46, i64 %3)
  %90 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %91 = ptrtoint ptr %90 to i64
  %92 = add i64 %91, 63
  %93 = urem i64 %92, 64
  %94 = sub i64 %92, %93
  %95 = inttoptr i64 %94 to ptr
  %96 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %90, 0
  %97 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %96, ptr %95, 1
  %98 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %97, i64 0, 2
  %99 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %98, i64 288, 3, 0
  %100 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %99, i64 1, 4, 0
  %101 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 3, 0
  %102 = mul i64 %101, 1
  %103 = mul i64 %102, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %104 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 1
  %105 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 2
  %106 = getelementptr float, ptr %104, i64 %105
  call void @llvm.memcpy.p0.p0.i64(ptr %95, ptr %106, i64 %103, i1 false)
  %107 = getelementptr float, ptr %95, i64 %59
  call void @llvm.memcpy.p0.p0.i64(ptr %107, ptr %89, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %108 = add i64 %55, 1
  br label %54

109:                                              ; preds = %54
  %110 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, 0
  ret ptr %110
}

define ptr @attn(ptr %0, ptr %1, ptr %2, i64 %3) {
  %5 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %6 = ptrtoint ptr %5 to i64
  %7 = add i64 %6, 63
  %8 = urem i64 %7, 64
  %9 = sub i64 %7, %8
  %10 = inttoptr i64 %9 to ptr
  br label %11

11:                                               ; preds = %14, %4
  %12 = phi i64 [ %16, %14 ], [ 0, %4 ]
  %13 = icmp slt i64 %12, 256
  br i1 %13, label %14, label %17

14:                                               ; preds = %11
  %15 = getelementptr float, ptr %10, i64 %12
  store float 0xFFF0000000000000, ptr %15, align 4
  %16 = add i64 %12, 1
  br label %11

17:                                               ; preds = %11
  %18 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %19 = ptrtoint ptr %18 to i64
  %20 = add i64 %19, 63
  %21 = urem i64 %20, 64
  %22 = sub i64 %20, %21
  %23 = inttoptr i64 %22 to ptr
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %18, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, ptr %23, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 0, 2
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 256, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %23, ptr %10, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 256), i1 false)
  %29 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %30 = ptrtoint ptr %29 to i64
  %31 = add i64 %30, 63
  %32 = urem i64 %31, 64
  %33 = sub i64 %31, %32
  %34 = inttoptr i64 %33 to ptr
  %35 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %36 = ptrtoint ptr %35 to i64
  %37 = add i64 %36, 63
  %38 = urem i64 %37, 64
  %39 = sub i64 %37, %38
  %40 = inttoptr i64 %39 to ptr
  %41 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %42 = ptrtoint ptr %41 to i64
  %43 = add i64 %42, 63
  %44 = urem i64 %43, 64
  %45 = sub i64 %43, %44
  %46 = inttoptr i64 %45 to ptr
  %47 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %48 = ptrtoint ptr %47 to i64
  %49 = add i64 %48, 63
  %50 = urem i64 %49, 64
  %51 = sub i64 %49, %50
  %52 = inttoptr i64 %51 to ptr
  %53 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %54 = ptrtoint ptr %53 to i64
  %55 = add i64 %54, 63
  %56 = urem i64 %55, 64
  %57 = sub i64 %55, %56
  %58 = inttoptr i64 %57 to ptr
  br label %59

59:                                               ; preds = %76, %17
  %60 = phi i64 [ %101, %76 ], [ 0, %17 ]
  %61 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %93, %76 ], [ %28, %17 ]
  %62 = icmp slt i64 %60, %3
  br i1 %62, label %63, label %102

63:                                               ; preds = %59
  %64 = mul i64 %60, 48
  %65 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1, ptr @dpu_program_9)
  call void @upmemrt_dpu_scatter(ptr %65, ptr %0, i64 4, i64 48, i64 6, i64 192, i64 0, ptr @scatter_map_6)
  %66 = getelementptr float, ptr %1, i64 %64
  call void @llvm.memcpy.p0.p0.i64(ptr %34, ptr %66, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  call void @upmemrt_dpu_scatter(ptr %65, ptr %34, i64 4, i64 48, i64 6, i64 192, i64 192, ptr @scatter_map_6)
  call void @upmemrt_dpu_scatter(ptr %65, ptr @__constant_8x6xf32, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  call void @upmemrt_dpu_launch(ptr %65)
  call void @upmemrt_dpu_gather(ptr %65, ptr %40, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  call void @upmemrt_dpu_free(ptr %65)
  call void @llvm.memcpy.p0.p0.i64(ptr %52, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %67

67:                                               ; preds = %70, %63
  %68 = phi i64 [ %75, %70 ], [ 0, %63 ]
  %69 = icmp slt i64 %68, 48
  br i1 %69, label %70, label %76

70:                                               ; preds = %67
  %71 = getelementptr float, ptr %40, i64 %68
  %72 = load float, ptr %71, align 4
  %73 = load float, ptr %52, align 4
  %74 = fadd float %72, %73
  store float %74, ptr %52, align 4
  %75 = add i64 %68, 1
  br label %67

76:                                               ; preds = %67
  %77 = load float, ptr %52, align 4
  store float %77, ptr %46, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %58, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %78 = load float, ptr %46, align 4
  %79 = load float, ptr %58, align 4
  %80 = fadd float %78, %79
  store float %80, ptr %58, align 4
  %81 = load float, ptr %58, align 4
  %82 = fdiv float %81, 0x401BB67AE0000000
  %83 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %84 = ptrtoint ptr %83 to i64
  %85 = add i64 %84, 63
  %86 = urem i64 %85, 64
  %87 = sub i64 %85, %86
  %88 = inttoptr i64 %87 to ptr
  %89 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %83, 0
  %90 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %89, ptr %88, 1
  %91 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %90, i64 0, 2
  %92 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %91, i64 256, 3, 0
  %93 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %92, i64 1, 4, 0
  %94 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 3, 0
  %95 = mul i64 %94, 1
  %96 = mul i64 %95, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %97 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 1
  %98 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 2
  %99 = getelementptr float, ptr %97, i64 %98
  call void @llvm.memcpy.p0.p0.i64(ptr %88, ptr %99, i64 %96, i1 false)
  %100 = getelementptr float, ptr %88, i64 %60
  store float %82, ptr %100, align 4
  %101 = add i64 %60, 1
  br label %59

102:                                              ; preds = %59
  %103 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %104 = ptrtoint ptr %103 to i64
  %105 = add i64 %104, 63
  %106 = urem i64 %105, 64
  %107 = sub i64 %105, %106
  %108 = inttoptr i64 %107 to ptr
  %109 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 3, 0
  %110 = mul i64 %109, 1
  %111 = mul i64 %110, ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64)
  %112 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 1
  %113 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, 2
  %114 = getelementptr float, ptr %112, i64 %113
  call void @llvm.memcpy.p0.p0.i64(ptr %108, ptr %114, i64 %111, i1 false)
  %115 = call ptr @softmax(ptr %108)
  %116 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %117 = ptrtoint ptr %116 to i64
  %118 = add i64 %117, 63
  %119 = urem i64 %118, 64
  %120 = sub i64 %118, %119
  %121 = inttoptr i64 %120 to ptr
  %122 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %123 = ptrtoint ptr %122 to i64
  %124 = add i64 %123, 63
  %125 = urem i64 %124, 64
  %126 = sub i64 %124, %125
  %127 = inttoptr i64 %126 to ptr
  %128 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %122, 0
  %129 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %128, ptr %127, 1
  %130 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %129, i64 0, 2
  %131 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %130, i64 48, 3, 0
  %132 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %131, i64 1, 4, 0
  call void @llvm.memcpy.p0.p0.i64(ptr %127, ptr %121, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  %133 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %134 = ptrtoint ptr %133 to i64
  %135 = add i64 %134, 63
  %136 = urem i64 %135, 64
  %137 = sub i64 %135, %136
  %138 = inttoptr i64 %137 to ptr
  %139 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 8) to i64), i64 64))
  %140 = ptrtoint ptr %139 to i64
  %141 = add i64 %140, 63
  %142 = urem i64 %141, 64
  %143 = sub i64 %141, %142
  %144 = inttoptr i64 %143 to ptr
  %145 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %146 = ptrtoint ptr %145 to i64
  %147 = add i64 %146, 63
  %148 = urem i64 %147, 64
  %149 = sub i64 %147, %148
  %150 = inttoptr i64 %149 to ptr
  br label %151

151:                                              ; preds = %155, %102
  %152 = phi i64 [ %181, %155 ], [ 0, %102 ]
  %153 = phi { ptr, ptr, i64, [1 x i64], [1 x i64] } [ %180, %155 ], [ %132, %102 ]
  %154 = icmp slt i64 %152, %3
  br i1 %154, label %155, label %182

155:                                              ; preds = %151
  %156 = mul i64 %152, 48
  %157 = getelementptr float, ptr %115, i64 %152
  %158 = load float, ptr %157, align 4
  %159 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1, ptr @dpu_program_10)
  %160 = getelementptr float, ptr %2, i64 %156
  call void @llvm.memcpy.p0.p0.i64(ptr %138, ptr %160, i64 mul (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 48), i1 false)
  call void @upmemrt_dpu_scatter(ptr %159, ptr %138, i64 4, i64 48, i64 6, i64 192, i64 0, ptr @scatter_map_6)
  store float %158, ptr %144, align 4
  %161 = getelementptr float, ptr %144, i32 1
  store float %158, ptr %161, align 4
  %162 = getelementptr float, ptr %144, i32 2
  store float %158, ptr %162, align 4
  %163 = getelementptr float, ptr %144, i32 3
  store float %158, ptr %163, align 4
  %164 = getelementptr float, ptr %144, i32 4
  store float %158, ptr %164, align 4
  %165 = getelementptr float, ptr %144, i32 5
  store float %158, ptr %165, align 4
  %166 = getelementptr float, ptr %144, i32 6
  store float %158, ptr %166, align 4
  %167 = getelementptr float, ptr %144, i32 7
  store float %158, ptr %167, align 4
  call void @upmemrt_dpu_scatter(ptr %159, ptr %144, i64 4, i64 8, i64 1, i64 32, i64 192, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %159, ptr @__constant_8x6xf32, i64 4, i64 48, i64 6, i64 192, i64 224, ptr @scatter_map_6)
  call void @upmemrt_dpu_launch(ptr %159)
  call void @upmemrt_dpu_gather(ptr %159, ptr %150, i64 4, i64 48, i64 6, i64 192, i64 224, ptr @scatter_map_6)
  call void @upmemrt_dpu_free(ptr %159)
  %168 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1, ptr @dpu_program_11)
  call void @upmemrt_dpu_scatter(ptr %168, ptr %150, i64 4, i64 48, i64 6, i64 192, i64 0, ptr @scatter_map_6)
  %169 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %153, 1
  call void @upmemrt_dpu_scatter(ptr %168, ptr %169, i64 4, i64 48, i64 6, i64 192, i64 192, ptr @scatter_map_6)
  call void @upmemrt_dpu_scatter(ptr %168, ptr @__constant_8x6xf32, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  call void @upmemrt_dpu_launch(ptr %168)
  %170 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 48) to i64), i64 64))
  %171 = ptrtoint ptr %170 to i64
  %172 = add i64 %171, 63
  %173 = urem i64 %172, 64
  %174 = sub i64 %172, %173
  %175 = inttoptr i64 %174 to ptr
  call void @upmemrt_dpu_gather(ptr %168, ptr %175, i64 4, i64 48, i64 6, i64 192, i64 384, ptr @scatter_map_6)
  %176 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %170, 0
  %177 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %176, ptr %175, 1
  %178 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %177, i64 0, 2
  %179 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %178, i64 48, 3, 0
  %180 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %179, i64 1, 4, 0
  call void @upmemrt_dpu_free(ptr %168)
  %181 = add i64 %152, 1
  br label %151

182:                                              ; preds = %151
  %183 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %153, 0
  ret ptr %183
}

define ptr @rmsnorm(ptr %0, ptr %1) {
  %3 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1, ptr @dpu_program_12)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 288, i64 18, i64 1152, i64 0, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %3, ptr %0, i64 4, i64 288, i64 18, i64 1152, i64 1152, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %3, ptr @__constant_16x18xf32, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_launch(ptr %3)
  %4 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %5 = ptrtoint ptr %4 to i64
  %6 = add i64 %5, 63
  %7 = urem i64 %6, 64
  %8 = sub i64 %6, %7
  %9 = inttoptr i64 %8 to ptr
  call void @upmemrt_dpu_gather(ptr %3, ptr %9, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_free(ptr %3)
  %10 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = urem i64 %12, 64
  %14 = sub i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %17 = ptrtoint ptr %16 to i64
  %18 = add i64 %17, 63
  %19 = urem i64 %18, 64
  %20 = sub i64 %18, %19
  %21 = inttoptr i64 %20 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %21, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %22

22:                                               ; preds = %25, %2
  %23 = phi i64 [ %30, %25 ], [ 0, %2 ]
  %24 = icmp slt i64 %23, 288
  br i1 %24, label %25, label %31

25:                                               ; preds = %22
  %26 = getelementptr float, ptr %9, i64 %23
  %27 = load float, ptr %26, align 4
  %28 = load float, ptr %21, align 4
  %29 = fadd float %27, %28
  store float %29, ptr %21, align 4
  %30 = add i64 %23, 1
  br label %22

31:                                               ; preds = %22
  %32 = load float, ptr %21, align 4
  store float %32, ptr %15, align 4
  %33 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %34 = ptrtoint ptr %33 to i64
  %35 = add i64 %34, 63
  %36 = urem i64 %35, 64
  %37 = sub i64 %35, %36
  %38 = inttoptr i64 %37 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %38, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %39 = load float, ptr %15, align 4
  %40 = load float, ptr %38, align 4
  %41 = fadd float %39, %40
  store float %41, ptr %38, align 4
  %42 = load float, ptr %38, align 4
  %43 = fdiv float %42, 2.880000e+02
  %44 = fadd float %43, 0x3EE4F8B580000000
  %45 = call float @llvm.sqrt.f32(float %44)
  %46 = fdiv float 1.000000e+00, %45
  %47 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1, ptr @dpu_program_13)
  call void @upmemrt_dpu_scatter(ptr %47, ptr %0, i64 4, i64 288, i64 18, i64 1152, i64 0, ptr @scatter_map_7)
  %48 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64), i64 64))
  %49 = ptrtoint ptr %48 to i64
  %50 = add i64 %49, 63
  %51 = urem i64 %50, 64
  %52 = sub i64 %50, %51
  %53 = inttoptr i64 %52 to ptr
  store float %46, ptr %53, align 4
  %54 = getelementptr float, ptr %53, i32 1
  store float %46, ptr %54, align 4
  %55 = getelementptr float, ptr %53, i32 2
  store float %46, ptr %55, align 4
  %56 = getelementptr float, ptr %53, i32 3
  store float %46, ptr %56, align 4
  %57 = getelementptr float, ptr %53, i32 4
  store float %46, ptr %57, align 4
  %58 = getelementptr float, ptr %53, i32 5
  store float %46, ptr %58, align 4
  %59 = getelementptr float, ptr %53, i32 6
  store float %46, ptr %59, align 4
  %60 = getelementptr float, ptr %53, i32 7
  store float %46, ptr %60, align 4
  %61 = getelementptr float, ptr %53, i32 8
  store float %46, ptr %61, align 4
  %62 = getelementptr float, ptr %53, i32 9
  store float %46, ptr %62, align 4
  %63 = getelementptr float, ptr %53, i32 10
  store float %46, ptr %63, align 4
  %64 = getelementptr float, ptr %53, i32 11
  store float %46, ptr %64, align 4
  %65 = getelementptr float, ptr %53, i32 12
  store float %46, ptr %65, align 4
  %66 = getelementptr float, ptr %53, i32 13
  store float %46, ptr %66, align 4
  %67 = getelementptr float, ptr %53, i32 14
  store float %46, ptr %67, align 4
  %68 = getelementptr float, ptr %53, i32 15
  store float %46, ptr %68, align 4
  call void @upmemrt_dpu_scatter(ptr %47, ptr %53, i64 4, i64 16, i64 1, i64 64, i64 1152, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %47, ptr @__constant_16x18xf32, i64 4, i64 288, i64 18, i64 1152, i64 1216, ptr @scatter_map_7)
  call void @upmemrt_dpu_launch(ptr %47)
  %69 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %70 = ptrtoint ptr %69 to i64
  %71 = add i64 %70, 63
  %72 = urem i64 %71, 64
  %73 = sub i64 %71, %72
  %74 = inttoptr i64 %73 to ptr
  call void @upmemrt_dpu_gather(ptr %47, ptr %74, i64 4, i64 288, i64 18, i64 1152, i64 1216, ptr @scatter_map_7)
  call void @upmemrt_dpu_free(ptr %47)
  %75 = call ptr @upmemrt_dpu_alloc(i32 1, i32 1, ptr @dpu_program_14)
  call void @upmemrt_dpu_scatter(ptr %75, ptr %74, i64 4, i64 288, i64 18, i64 1152, i64 0, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %75, ptr %1, i64 4, i64 288, i64 18, i64 1152, i64 1152, ptr @scatter_map_7)
  call void @upmemrt_dpu_scatter(ptr %75, ptr @__constant_16x18xf32, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_launch(ptr %75)
  %76 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 288) to i64), i64 64))
  %77 = ptrtoint ptr %76 to i64
  %78 = add i64 %77, 63
  %79 = urem i64 %78, 64
  %80 = sub i64 %78, %79
  %81 = inttoptr i64 %80 to ptr
  call void @upmemrt_dpu_gather(ptr %75, ptr %81, i64 4, i64 288, i64 18, i64 1152, i64 2304, ptr @scatter_map_7)
  call void @upmemrt_dpu_free(ptr %75)
  ret ptr %76
}

define ptr @softmax(ptr %0) {
  %2 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
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
  call void @llvm.memcpy.p0.p0.i64(ptr %13, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %14

14:                                               ; preds = %17, %1
  %15 = phi i64 [ %22, %17 ], [ 0, %1 ]
  %16 = icmp slt i64 %15, 256
  br i1 %16, label %17, label %23

17:                                               ; preds = %14
  %18 = getelementptr float, ptr %0, i64 %15
  %19 = load float, ptr %18, align 4
  %20 = load float, ptr %13, align 4
  %21 = call float @llvm.maximum.f32(float %19, float %20)
  store float %21, ptr %13, align 4
  %22 = add i64 %15, 1
  br label %14

23:                                               ; preds = %14
  %24 = load float, ptr %13, align 4
  store float %24, ptr %7, align 4
  %25 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %26 = ptrtoint ptr %25 to i64
  %27 = add i64 %26, 63
  %28 = urem i64 %27, 64
  %29 = sub i64 %27, %28
  %30 = inttoptr i64 %29 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %30, ptr @__constant_xf32, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %31 = load float, ptr %7, align 4
  %32 = load float, ptr %30, align 4
  %33 = call float @llvm.maximum.f32(float %31, float %32)
  store float %33, ptr %30, align 4
  %34 = load float, ptr %30, align 4
  %35 = call ptr @upmemrt_dpu_alloc(i32 1, i32 8, ptr @dpu_program_15)
  call void @upmemrt_dpu_scatter(ptr %35, ptr %0, i64 4, i64 256, i64 2, i64 128, i64 0, ptr @scatter_map_8)
  %36 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64), i64 64))
  %37 = ptrtoint ptr %36 to i64
  %38 = add i64 %37, 63
  %39 = urem i64 %38, 64
  %40 = sub i64 %38, %39
  %41 = inttoptr i64 %40 to ptr
  store float %34, ptr %41, align 4
  %42 = getelementptr float, ptr %41, i32 1
  store float %34, ptr %42, align 4
  %43 = getelementptr float, ptr %41, i32 2
  store float %34, ptr %43, align 4
  %44 = getelementptr float, ptr %41, i32 3
  store float %34, ptr %44, align 4
  %45 = getelementptr float, ptr %41, i32 4
  store float %34, ptr %45, align 4
  %46 = getelementptr float, ptr %41, i32 5
  store float %34, ptr %46, align 4
  %47 = getelementptr float, ptr %41, i32 6
  store float %34, ptr %47, align 4
  %48 = getelementptr float, ptr %41, i32 7
  store float %34, ptr %48, align 4
  %49 = getelementptr float, ptr %41, i32 8
  store float %34, ptr %49, align 4
  %50 = getelementptr float, ptr %41, i32 9
  store float %34, ptr %50, align 4
  %51 = getelementptr float, ptr %41, i32 10
  store float %34, ptr %51, align 4
  %52 = getelementptr float, ptr %41, i32 11
  store float %34, ptr %52, align 4
  %53 = getelementptr float, ptr %41, i32 12
  store float %34, ptr %53, align 4
  %54 = getelementptr float, ptr %41, i32 13
  store float %34, ptr %54, align 4
  %55 = getelementptr float, ptr %41, i32 14
  store float %34, ptr %55, align 4
  %56 = getelementptr float, ptr %41, i32 15
  store float %34, ptr %56, align 4
  call void @upmemrt_dpu_scatter(ptr %35, ptr %41, i64 4, i64 16, i64 0, i64 64, i64 128, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %35, ptr @__constant_128x2xf32, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_launch(ptr %35)
  %57 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %58 = ptrtoint ptr %57 to i64
  %59 = add i64 %58, 63
  %60 = urem i64 %59, 64
  %61 = sub i64 %59, %60
  %62 = inttoptr i64 %61 to ptr
  call void @upmemrt_dpu_gather(ptr %35, ptr %62, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_free(ptr %35)
  %63 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %64 = ptrtoint ptr %63 to i64
  %65 = add i64 %64, 63
  %66 = urem i64 %65, 64
  %67 = sub i64 %65, %66
  %68 = inttoptr i64 %67 to ptr
  br label %69

69:                                               ; preds = %72, %23
  %70 = phi i64 [ %77, %72 ], [ 0, %23 ]
  %71 = icmp slt i64 %70, 256
  br i1 %71, label %72, label %78

72:                                               ; preds = %69
  %73 = getelementptr float, ptr %62, i64 %70
  %74 = load float, ptr %73, align 4
  %75 = call float @llvm.exp.f32(float %74)
  %76 = getelementptr float, ptr %68, i64 %70
  store float %75, ptr %76, align 4
  %77 = add i64 %70, 1
  br label %69

78:                                               ; preds = %69
  %79 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %80 = ptrtoint ptr %79 to i64
  %81 = add i64 %80, 63
  %82 = urem i64 %81, 64
  %83 = sub i64 %81, %82
  %84 = inttoptr i64 %83 to ptr
  %85 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %86 = ptrtoint ptr %85 to i64
  %87 = add i64 %86, 63
  %88 = urem i64 %87, 64
  %89 = sub i64 %87, %88
  %90 = inttoptr i64 %89 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %90, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  br label %91

91:                                               ; preds = %94, %78
  %92 = phi i64 [ %99, %94 ], [ 0, %78 ]
  %93 = icmp slt i64 %92, 256
  br i1 %93, label %94, label %100

94:                                               ; preds = %91
  %95 = getelementptr float, ptr %68, i64 %92
  %96 = load float, ptr %95, align 4
  %97 = load float, ptr %90, align 4
  %98 = fadd float %96, %97
  store float %98, ptr %90, align 4
  %99 = add i64 %92, 1
  br label %91

100:                                              ; preds = %91
  %101 = load float, ptr %90, align 4
  store float %101, ptr %84, align 4
  %102 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i64 64))
  %103 = ptrtoint ptr %102 to i64
  %104 = add i64 %103, 63
  %105 = urem i64 %104, 64
  %106 = sub i64 %104, %105
  %107 = inttoptr i64 %106 to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %107, ptr @__constant_xf32_0, i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1) to i64), i1 false)
  %108 = load float, ptr %84, align 4
  %109 = load float, ptr %107, align 4
  %110 = fadd float %108, %109
  store float %110, ptr %107, align 4
  %111 = load float, ptr %107, align 4
  %112 = call ptr @upmemrt_dpu_alloc(i32 1, i32 8, ptr @dpu_program_16)
  call void @upmemrt_dpu_scatter(ptr %112, ptr %68, i64 4, i64 256, i64 2, i64 128, i64 0, ptr @scatter_map_8)
  %113 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 16) to i64), i64 64))
  %114 = ptrtoint ptr %113 to i64
  %115 = add i64 %114, 63
  %116 = urem i64 %115, 64
  %117 = sub i64 %115, %116
  %118 = inttoptr i64 %117 to ptr
  store float %111, ptr %118, align 4
  %119 = getelementptr float, ptr %118, i32 1
  store float %111, ptr %119, align 4
  %120 = getelementptr float, ptr %118, i32 2
  store float %111, ptr %120, align 4
  %121 = getelementptr float, ptr %118, i32 3
  store float %111, ptr %121, align 4
  %122 = getelementptr float, ptr %118, i32 4
  store float %111, ptr %122, align 4
  %123 = getelementptr float, ptr %118, i32 5
  store float %111, ptr %123, align 4
  %124 = getelementptr float, ptr %118, i32 6
  store float %111, ptr %124, align 4
  %125 = getelementptr float, ptr %118, i32 7
  store float %111, ptr %125, align 4
  %126 = getelementptr float, ptr %118, i32 8
  store float %111, ptr %126, align 4
  %127 = getelementptr float, ptr %118, i32 9
  store float %111, ptr %127, align 4
  %128 = getelementptr float, ptr %118, i32 10
  store float %111, ptr %128, align 4
  %129 = getelementptr float, ptr %118, i32 11
  store float %111, ptr %129, align 4
  %130 = getelementptr float, ptr %118, i32 12
  store float %111, ptr %130, align 4
  %131 = getelementptr float, ptr %118, i32 13
  store float %111, ptr %131, align 4
  %132 = getelementptr float, ptr %118, i32 14
  store float %111, ptr %132, align 4
  %133 = getelementptr float, ptr %118, i32 15
  store float %111, ptr %133, align 4
  call void @upmemrt_dpu_scatter(ptr %112, ptr %118, i64 4, i64 16, i64 0, i64 64, i64 128, ptr @scatter_map_0)
  call void @upmemrt_dpu_scatter(ptr %112, ptr @__constant_128x2xf32, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_launch(ptr %112)
  %134 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 256) to i64), i64 64))
  %135 = ptrtoint ptr %134 to i64
  %136 = add i64 %135, 63
  %137 = urem i64 %136, 64
  %138 = sub i64 %136, %137
  %139 = inttoptr i64 %138 to ptr
  call void @upmemrt_dpu_gather(ptr %112, ptr %139, i64 4, i64 256, i64 2, i64 128, i64 192, ptr @scatter_map_8)
  call void @upmemrt_dpu_free(ptr %112)
  ret ptr %134
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.pow.f32(float, float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.cos.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sin.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
