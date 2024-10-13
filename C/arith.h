#include<stdio.h>
#include<stdint.h>

#ifndef arith_H

#define arith_H

typedef uint16_t bf16_t;

bf16_t fp32_to_bf16(float s);
bf16_t bf16_add(bf16_t a, bf16_t b);
bf16_t bf16_mul(bf16_t a, bf16_t b);
bf16_t bf16_div(bf16_t a, bf16_t b);
bf16_t bf16_abs(bf16_t f);
float bf16_to_float(bf16_t bf);
void swap(bf16_t *a, bf16_t *b);
#endif  