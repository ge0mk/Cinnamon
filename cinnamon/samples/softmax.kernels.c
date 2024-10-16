// UPMEM-TRANSLATE: COMPILE_softmax:16:softmax;COMPILE_softmax_0:16:softmax_0;

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef COMPILE_softmax
void softmax() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 64;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[16];
  int32_t v6 = v1 + 1024;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[16];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 16 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 16; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 - v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 16 * sizeof(float));
  return;
}
#endif

#ifdef COMPILE_softmax_0
void softmax_0() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = me();
  int32_t v3 = v2 * 64;
  int32_t v4 = v1 + v3;
  __dma_aligned float v5[16];
  int32_t v6 = v1 + 1024;
  __dma_aligned float v7[2];
  int32_t v8 = v6 + 64;
  int32_t v9 = v8 + v3;
  __dma_aligned float v10[16];
  mram_read((const __mram_ptr float*)v4, (float*)v5, 16 * sizeof(float));
  mram_read((const __mram_ptr float*)v6, (float*)v7, 2 * sizeof(float));
  for (int32_t v11 = 0; v11 < 16; v11 += 1) {
    float v12 = v5[v11];
    float v13 = v7[0];
    float v14 = v12 / v13;
    v10[v11] = v14;
  }
  mram_write((const float*)v10, (__mram_ptr float*)v9, 16 * sizeof(float));
  return;
}
#endif

BARRIER_INIT(my_barrier, NR_TASKLETS);

int main(void) {
  barrier_wait(&my_barrier);
  mem_reset();
#ifdef COMPILE_softmax
  softmax();
#endif
#ifdef COMPILE_softmax_0
  softmax_0();
#endif
  mem_reset();
  return 0;
}
