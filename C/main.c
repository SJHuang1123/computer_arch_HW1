#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"arith.h"

bf16_t determinant(bf16_t **matrix)
{
    bf16_t a = matrix[0][0];
    bf16_t b = matrix[0][1];
    bf16_t c = matrix[1][0];
    bf16_t d = matrix[1][1];
    return bf16_add(bf16_mul(a, d), bf16_mul(b, c) ^ 0x8000);
}

bf16_t **inv(bf16_t **matrix)
{
    bf16_t det = determinant(matrix);

    if(bf16_abs(det) > 1e-7)
    {
        swap(*matrix, *(matrix + 1) + 1);
        matrix[0][1] = matrix[0][1] ^ 0x8000;
        matrix[1][0] = matrix[1][0] ^ 0x8000;

        matrix[0][0] = bf16_div(matrix[0][0], det);
        matrix[0][1] = bf16_div(matrix[0][1], det);
        matrix[1][0] = bf16_div(matrix[1][0], det);
        matrix[1][1] = bf16_div(matrix[1][1], det);
        return matrix;
    }
    return NULL;
}

int main(void)
{
    bf16_t **test = malloc(2 * sizeof(bf16_t *));
    *test = malloc(2 * sizeof(bf16_t));
    *(test + 1) = malloc(2 * sizeof(bf16_t));
    test[0][0] = fp32_to_bf16(9.0f); test[0][1] = fp32_to_bf16(5.0f); test[1][0] = fp32_to_bf16(2.0f); test[1][1] = fp32_to_bf16(7.0f);
    for(int i = 0; i < 2; i ++)
    {
        for(int j = 0; j < 2; j++)
            printf("%f ", bf16_to_float(test[i][j]));
        printf("\n");
    }
    printf("\n");
    bf16_t ans = determinant(test);
    printf("Determinant:\n%f\n", bf16_to_float(ans));
    if(inv(test) == NULL) printf("matrix is singular\n");
    else
    {
        for(int i = 0; i < 2; i ++)
        {
            free(test[i]);
        }
        free(test);
    }
    return 0;
}