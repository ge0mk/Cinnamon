// UPMEM-TRANSLATE: COMPILE_forward:8:forward;COMPILE_forward_3:8:forward_3;COMPILE_forward_6:8:forward_6;COMPILE_forward_8:16:forward_8;COMPILE_attn:16:attn;COMPILE_attn_9:16:attn_9;COMPILE_attn_10:16:attn_10;COMPILE_softmax:16:softmax;COMPILE_softmax_13:16:softmax_13;

#include "dpu_lib.h"

void forward();
void forward_3();
void forward_6();
void forward_8();
void attn();
void attn_9();
void attn_10();
void softmax();
void softmax_13();

int main(void) {
  init_tasklet();
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
#ifdef COMPILE_attn_10
  attn_10();
#endif
#ifdef COMPILE_softmax
  softmax();
#endif
#ifdef COMPILE_softmax_13
  softmax_13();
#endif
  return 0;
}

void forward() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 288;
  float * v3 = (float*) dpu_wram_alloc(288 * sizeof(float));
  int32_t v4 = 2304;
  int32_t v5 = v1 + v4;
  float * v6 = (float*) dpu_wram_alloc(288 * sizeof(float));
  int32_t v7 = v5 + v4;
  int32_t v8 = 1;
  float * v9 = (float*) dpu_wram_alloc(1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v6, v2 * sizeof(float), v5 * sizeof(float));
  int32_t v10 = 0;
  for (int32_t v11 = v10; v11 < v2; v11 += v8) {
    float v12 = v3[v11];
    float v13 = v6[v11];
    float v14 = v9[0];
    float v15 = v12 * v13;
    float v16 = v15 + v14;
    v9[0] = v16;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v9, v8 * sizeof(float), v7 * sizeof(float));
  ;
}
void forward_3() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 6;
  float * v3 = (float*) dpu_wram_alloc(6 * sizeof(float));
  int32_t v4 = 48;
  int32_t v5 = v1 + v4;
  float * v6 = (float*) dpu_wram_alloc(6 * sizeof(float));
  int32_t v7 = v5 + v4;
  float * v8 = (float*) dpu_wram_alloc(6 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v6, v2 * sizeof(float), v5 * sizeof(float));
  int32_t v9 = 0;
  int32_t v10 = 1;
  for (int32_t v11 = v9; v11 < v2; v11 += v10) {
    float v12 = v3[v11];
    float v13 = v6[v11];
    float v14 = v12 + v13;
    v8[v11] = v14;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v8, v2 * sizeof(float), v7 * sizeof(float));
  ;
}
void forward_6() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 768;
  float * v3 = (float*) dpu_wram_alloc(768 * sizeof(float));
  int32_t v4 = 6144;
  int32_t v5 = v1 + v4;
  float * v6 = (float*) dpu_wram_alloc(768 * sizeof(float));
  int32_t v7 = v5 + v4;
  int32_t v8 = 1;
  float * v9 = (float*) dpu_wram_alloc(1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v6, v2 * sizeof(float), v5 * sizeof(float));
  int32_t v10 = 0;
  for (int32_t v11 = v10; v11 < v2; v11 += v8) {
    float v12 = v3[v11];
    float v13 = v6[v11];
    float v14 = v9[0];
    float v15 = v12 * v13;
    float v16 = v15 + v14;
    v9[0] = v16;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v9, v8 * sizeof(float), v7 * sizeof(float));
  ;
}
void forward_8() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 288;
  float * v3 = (float*) dpu_wram_alloc(288 * sizeof(float));
  int32_t v4 = 4608;
  int32_t v5 = v1 + v4;
  float * v6 = (float*) dpu_wram_alloc(288 * sizeof(float));
  int32_t v7 = v5 + v4;
  int32_t v8 = 1;
  float * v9 = (float*) dpu_wram_alloc(1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v6, v2 * sizeof(float), v5 * sizeof(float));
  int32_t v10 = 0;
  for (int32_t v11 = v10; v11 < v2; v11 += v8) {
    float v12 = v3[v11];
    float v13 = v6[v11];
    float v14 = v9[0];
    float v15 = v12 * v13;
    float v16 = v15 + v14;
    v9[0] = v16;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v9, v8 * sizeof(float), v7 * sizeof(float));
  ;
}
void attn() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 3;
  float * v3 = (float*) dpu_wram_alloc(3 * sizeof(float));
  int32_t v4 = 48;
  int32_t v5 = v1 + v4;
  float * v6 = (float*) dpu_wram_alloc(3 * sizeof(float));
  int32_t v7 = v5 + v4;
  float * v8 = (float*) dpu_wram_alloc(3 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v6, v2 * sizeof(float), v5 * sizeof(float));
  int32_t v9 = 0;
  int32_t v10 = 1;
  for (int32_t v11 = v9; v11 < v2; v11 += v10) {
    float v12 = v3[v11];
    float v13 = v6[v11];
    float v14 = v12 * v13;
    v8[v11] = v14;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v8, v2 * sizeof(float), v7 * sizeof(float));
  ;
}
void attn_9() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 3;
  float * v3 = (float*) dpu_wram_alloc(3 * sizeof(float));
  int32_t v4 = 48;
  int32_t v5 = v1 + v4;
  int32_t v6 = 1;
  float * v7 = (float*) dpu_wram_alloc(1 * sizeof(float));
  int32_t v8 = 16;
  int32_t v9 = v5 + v8;
  float * v10 = (float*) dpu_wram_alloc(3 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v7, v6 * sizeof(float), v5 * sizeof(float));
  int32_t v11 = 0;
  for (int32_t v12 = v11; v12 < v2; v12 += v6) {
    float v13 = v3[v12];
    float v14 = v7[0];
    float v15 = v13 * v14;
    v10[v12] = v15;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v10, v2 * sizeof(float), v9 * sizeof(float));
  ;
}
void attn_10() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 3;
  float * v3 = (float*) dpu_wram_alloc(3 * sizeof(float));
  int32_t v4 = 48;
  int32_t v5 = v1 + v4;
  float * v6 = (float*) dpu_wram_alloc(3 * sizeof(float));
  int32_t v7 = v5 + v4;
  float * v8 = (float*) dpu_wram_alloc(3 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v6, v2 * sizeof(float), v5 * sizeof(float));
  int32_t v9 = 0;
  int32_t v10 = 1;
  for (int32_t v11 = v9; v11 < v2; v11 += v10) {
    float v12 = v3[v11];
    float v13 = v6[v11];
    float v14 = v12 + v13;
    v8[v11] = v14;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v8, v2 * sizeof(float), v7 * sizeof(float));
  ;
}
void softmax() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 2;
  float * v3 = (float*) dpu_wram_alloc(2 * sizeof(float));
  int32_t v4 = 32;
  int32_t v5 = v1 + v4;
  int32_t v6 = 1;
  float * v7 = (float*) dpu_wram_alloc(1 * sizeof(float));
  int32_t v8 = 16;
  int32_t v9 = v5 + v8;
  float * v10 = (float*) dpu_wram_alloc(2 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v7, v6 * sizeof(float), v5 * sizeof(float));
  int32_t v11 = 0;
  for (int32_t v12 = v11; v12 < v2; v12 += v6) {
    float v13 = v3[v12];
    float v14 = v7[0];
    float v15 = v13 - v14;
    v10[v12] = v15;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v10, v2 * sizeof(float), v9 * sizeof(float));
  ;
}
void softmax_13() {
  int32_t v1 = (uint32_t) DPU_MRAM_HEAP_POINTER;
  int32_t v2 = 2;
  float * v3 = (float*) dpu_wram_alloc(2 * sizeof(float));
  int32_t v4 = 32;
  int32_t v5 = v1 + v4;
  int32_t v6 = 1;
  float * v7 = (float*) dpu_wram_alloc(1 * sizeof(float));
  int32_t v8 = 16;
  int32_t v9 = v5 + v8;
  float * v10 = (float*) dpu_wram_alloc(2 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v3, v2 * sizeof(float), v1 * sizeof(float));
  dpu_memcpy(MRAMToWRAM, (int*)v7, v6 * sizeof(float), v5 * sizeof(float));
  int32_t v11 = 0;
  for (int32_t v12 = v11; v12 < v2; v12 += v6) {
    float v13 = v3[v12];
    float v14 = v7[0];
    float v15 = v13 / v14;
    v10[v12] = v15;
  }
  dpu_memcpy(WRAMToMRAM, (int*)v10, v2 * sizeof(float), v9 * sizeof(float));
  ;
}

