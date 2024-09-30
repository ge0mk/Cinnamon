// UPMEM-TRANSLATE: COMPILE_forward:8:forward;COMPILE_forward_3:8:forward_3;COMPILE_forward_6:8:forward_6;COMPILE_forward_8:16:forward_8;COMPILE_attn:8:attn;COMPILE_attn_9:8:attn_9;COMPILE_rmsnorm:16:rmsnorm;COMPILE_rmsnorm_11:16:rmsnorm_11;COMPILE_softmax:16:softmax;COMPILE_softmax_13:16:softmax_13;

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef COMPILE_forward
void forward() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1152;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[288];
  int32_t v6 = v1 + 9216;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[288];
  int32_t v9 = v6 + 9216;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 288 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 288 * sizeof(float));
  for (int32_t v13 = 0; v13 < 288; v13 += 1) {
    float v14 = v5[v13];
    float v15 = v8[v13];
    float v16 = v12[0];
    float v17 = v14 * v15;
    float v18 = v17 + v16;
    v12[0] = v18;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_forward_3
void forward_3() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 24;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[6];
  int32_t v6 = v1 + 192;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[6];
  int32_t v9 = v6 + 192;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[6];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 6 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 6 * sizeof(float));
  for (int32_t v12 = 0; v12 < 6; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 + v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 6 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_forward_6
void forward_6() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 3072;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[768];
  int32_t v6 = v1 + 24576;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[768];
  int32_t v9 = v6 + 24576;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 512 * sizeof(float));
  mram_read((const __mram_ptr float*)v4 + 512, (float*)v5 + 512, 256 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 512 * sizeof(float));
  mram_read((const __mram_ptr float*)v7 + 512, (float*)v8 + 512, 256 * sizeof(float));
  for (int32_t v13 = 0; v13 < 768; v13 += 1) {
    float v14 = v5[v13];
    float v15 = v8[v13];
    float v16 = v12[0];
    float v17 = v14 * v15;
    float v18 = v17 + v16;
    v12[0] = v18;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_forward_8
void forward_8() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 1152;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[288];
  int32_t v6 = v1 + 18432;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[288];
  int32_t v9 = v6 + 18432;
  int32_t v10 = v2 * 8;
  int32_t v11 = v9 + v10;
  __dma_aligned float v12[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 288 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 288 * sizeof(float));
  for (int32_t v13 = 0; v13 < 288; v13 += 1) {
    float v14 = v5[v13];
    float v15 = v8[v13];
    float v16 = v12[0];
    float v17 = v14 * v15;
    float v18 = v17 + v16;
    v12[0] = v18;
  }
  mram_write((const float*)v12, (__mram_ptr float*)v11, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_attn
void attn() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 24;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[6];
  int32_t v6 = v1 + 192;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[6];
  int32_t v9 = v6 + 192;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[6];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 6 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 6 * sizeof(float));
  for (int32_t v12 = 0; v12 < 6; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 6 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_attn_9
void attn_9() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 24;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[6];
  int32_t v6 = v1 + 192;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 32;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[6];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 6 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 6; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 6 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm
void rmsnorm() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 72;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[18];
  int32_t v6 = v1 + 1152;
  int32_t v7 = v6 + v3;
  __dma_aligned float v8[18];
  int32_t v9 = v6 + 1152;
  int32_t v10 = v9 + v3;
  __dma_aligned float v11[18];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 18 * sizeof(float));
  mram_read((const __mram_ptr float*)v7, (float*)v8, 18 * sizeof(float));
  for (int32_t v12 = 0; v12 < 18; v12 += 1) {
    float v13 = v5[v12];
    float v14 = v8[v12];
    float v15 = v13 * v14;
    v11[v12] = v15;
  }
  mram_write((const float*)v11, (__mram_ptr float*)v10, 18 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_rmsnorm_11
void rmsnorm_11() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 72;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[18];
  int32_t v6 = v1 + 1152;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[18];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 18 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 18; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 * v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 18 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax
void softmax() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 8;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[2];
  int32_t v6 = v1 + 128;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 2 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 2; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 - v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 2 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_13
void softmax_13() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 8;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[2];
  int32_t v6 = v1 + 128;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[2];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 2 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 2; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 / v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 2 * sizeof(float));
  return;
}
#endif

BARRIER_INIT(my_barrier, NR_TASKLETS);

int main(void) {
  barrier_wait(&my_barrier);
  mem_reset();
#ifdef COMPILE_forward
  forward();
#endif
#ifdef COMPILE_forward_3
  forward_3();
#endif
#ifdef COMPILE_forward_6
  forward_6();
#endif
#ifdef COMPILE_forward_8
  forward_8();
#endif
#ifdef COMPILE_attn
  attn();
#endif
#ifdef COMPILE_attn_9
  attn_9();
#endif
#ifdef COMPILE_rmsnorm
  rmsnorm();
#endif
#ifdef COMPILE_rmsnorm_11
  rmsnorm_11();
#endif
#ifdef COMPILE_softmax
  softmax();
#endif
#ifdef COMPILE_softmax_13
  softmax_13();
#endif
  mem_reset();
  return 0;
}
