#include<stdio.h>
#include<stdint.h>
#include"arith.h"

bf16_t bf16_abs(bf16_t f)
{
    f = f & 0x7fff;
    return f;
}

void swap(bf16_t *a, bf16_t *b)
{
    bf16_t temp = *a;
    *a = *b;
    *b = temp;
}

bf16_t fp32_to_bf16(float s)
{
    bf16_t h;
    union {
        float f;
        uint32_t i;
    } u = {.f = s};
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* NaN */
        h = (u.i >> 16) | 64;         /* force to quiet */
        return h;                                                                                                                                             
    }
    h = (u.i + (0x7fff + ((u.i >> 0x10) & 1))) >> 0x10;
    return h;
}

float bf16_to_float(uint16_t bf)
{
    uint32_t fp32 = ((uint32_t)bf) << 16;  
    return *((float*)&fp32);               
}

uint32_t get_sign(bf16_t num) {
    return (num >> 15) & 0x1;
}

uint32_t get_exponent(bf16_t num) {
    return (num >> 7) & 0xFF;
}

uint32_t get_mantissa(bf16_t num) {
    return num & 0x7F;
}

bf16_t assemble_bf16(uint32_t sign, uint32_t exponent, uint32_t mantissa) {
    return (sign << 15) | (exponent << 7) | (mantissa & 0x7F);
}


bf16_t bf16_add(bf16_t a, bf16_t b) {
    uint32_t sign_a = get_sign(a);
    uint32_t sign_b = get_sign(b);
    int32_t exp_a = get_exponent(a);
    int32_t exp_b = get_exponent(b);
    uint32_t mantissa_a = get_mantissa(a);
    uint32_t mantissa_b = get_mantissa(b);

    // Handle special cases (NaN, infinity, zero)
    if (exp_a == 0xFF) return a;  // a is NaN or infinity
    if (exp_b == 0xFF) return b;  // b is NaN or infinity
    if (exp_a == 0 && mantissa_a == 0) return b;  // a is zero
    if (exp_b == 0 && mantissa_b == 0) return a;  // b is zero

    // Restore implicit leading 1
    mantissa_a |= 0x80;
    mantissa_b |= 0x80;

    // Align exponents
    int32_t exp_diff = exp_a - exp_b;
    if (exp_diff > 0) {
        // Shift mantissa_b right
        if (exp_diff > 8) {  // Shift amount exceeds mantissa bits
            mantissa_b = 0;
        } else {
            mantissa_b >>= exp_diff;
        }
        exp_b = exp_a;
    } else if (exp_diff < 0) {
        // Shift mantissa_a right
        exp_diff = -exp_diff;
        if (exp_diff > 8) {
            mantissa_a = 0;
        } else {
            mantissa_a >>= exp_diff;
        }
        exp_a = exp_b;
    }

    // Perform addition or subtraction
    uint32_t result_sign = sign_a;
    int32_t result_mantissa;
    if (sign_a == sign_b) {
        result_mantissa = mantissa_a + mantissa_b;
    } else {
        if (mantissa_a >= mantissa_b) {
            result_mantissa = mantissa_a - mantissa_b;
        } else {
            result_mantissa = mantissa_b - mantissa_a;
            result_sign = sign_b;
        }
    }

    // Handle zero result
    if (result_mantissa == 0) {
        return assemble_bf16(0, 0, 0);
    }

    // Normalize result mantissa
    if (result_mantissa & 0x100) {  // Mantissa overflow (bit 8 is set)
        result_mantissa >>= 1;
        exp_a++;
    } else {
        while ((result_mantissa & 0x80) == 0) {  // Leading bit not in position
            result_mantissa <<= 1;
            exp_a--;
        }
    }

    // Handle exponent overflow/underflow
    if (exp_a >= 0xFF) {
        // Exponent overflow, return infinity
        return assemble_bf16(result_sign, 0xFF, 0);
    } else if (exp_a <= 0) {
        // Exponent underflow, return zero
        return assemble_bf16(0, 0, 0);
    }

    // Remove the implicit leading 1
    uint32_t mantissa = result_mantissa & 0x7F;

    return assemble_bf16(result_sign, exp_a, mantissa);
}

bf16_t bf16_mul(bf16_t a, bf16_t b) {
    uint32_t sign_a = get_sign(a);
    uint32_t sign_b = get_sign(b);
    int32_t exp_a = get_exponent(a);
    int32_t exp_b = get_exponent(b);
    uint32_t mantissa_a = get_mantissa(a);
    uint32_t mantissa_b = get_mantissa(b);

    // Handle special cases: zero, infinity, NaN
    if (exp_a == 0xFF) {
        // a is NaN or infinity
        if (mantissa_a != 0) {
            // a is NaN
            return assemble_bf16(0, 0xFF, 1); // Return NaN
        }
        if ((exp_b == 0 && mantissa_b == 0)) {
            // infinity * zero = NaN
            return assemble_bf16(0, 0xFF, 1); // Return NaN
        }
        // a is infinity
        return assemble_bf16(sign_a ^ sign_b, 0xFF, 0);
    }
    if (exp_b == 0xFF) {
        // b is NaN or infinity
        if (mantissa_b != 0) {
            // b is NaN
            return assemble_bf16(0, 0xFF, 1); // Return NaN
        }
        if ((exp_a == 0 && mantissa_a == 0)) {
            // zero * infinity = NaN
            return assemble_bf16(0, 0xFF, 1); // Return NaN
        }
        // b is infinity
        return assemble_bf16(sign_a ^ sign_b, 0xFF, 0);
    }
    if ((exp_a == 0 && mantissa_a == 0) || (exp_b == 0 && mantissa_b == 0)) {
        // One of the operands is zero, return zero
        return assemble_bf16(sign_a ^ sign_b, 0, 0);
    }

    // Restore implicit leading 1 for normalized numbers
    if (exp_a != 0) {
        mantissa_a |= 0x80;
    } else {
        // Subnormal number, adjust exponent
        exp_a = 1;
        // For subnormals, the leading bit is not set
    }
    if (exp_b != 0) {
        mantissa_b |= 0x80;
    } else {
        // Subnormal number, adjust exponent
        exp_b = 1;
    }

    // Compute the sign of the result
    uint32_t result_sign = sign_a ^ sign_b;

    // Multiply mantissas: 8 bits * 8 bits = 16 bits
    uint32_t mantissa_product = mantissa_a * mantissa_b; // Up to 0xFE01

    // Initial exponent calculation
    int32_t result_exp = exp_a + exp_b - 127;

    // Normalize the mantissa product
    uint32_t result_mantissa = mantissa_product;

    // Determine normalization and rounding
    if (result_mantissa & 0x8000) { // Bit 15 is set
        // Shift right by 8 to get the mantissa in [0x80, 0xFF]
        uint32_t rounding_bit = (result_mantissa >> 7) & 1;
        result_mantissa >>= 8;
        result_exp += 1;
        // Apply rounding
        if (rounding_bit) {
            result_mantissa += 1;
            if (result_mantissa == 0x100) { // Mantissa overflow after rounding
                result_mantissa >>= 1;
                result_exp += 1;
            }
        }
    } else {
        // Shift right by 7 to get the mantissa in [0x80, 0xFF]
        uint32_t rounding_bit = (result_mantissa >> 6) & 1;
        result_mantissa >>= 7;
        // Apply rounding
        if (rounding_bit) {
            result_mantissa += 1;
            if (result_mantissa == 0x100) { // Mantissa overflow after rounding
                result_mantissa >>= 1;
                result_exp += 1;
            }
        }
    }

    // Handle exponent overflow/underflow
    if (result_exp >= 0xFF) {
        // Exponent overflow, return infinity
        return assemble_bf16(result_sign, 0xFF, 0);
    } else if (result_exp <= 0) {
        // Exponent underflow, return zero
        return assemble_bf16(0, 0, 0);
    }

    // Remove the implicit leading 1
    uint32_t mantissa = result_mantissa & 0x7F;

    return assemble_bf16(result_sign, result_exp, mantissa);
}


bf16_t bf16_div(bf16_t a, bf16_t b) {
    uint32_t sign_a = get_sign(a);
    uint32_t sign_b = get_sign(b);
    int32_t exp_a = get_exponent(a);
    int32_t exp_b = get_exponent(b);
    uint32_t mantissa_a = get_mantissa(a) | 0x80; // Restore implicit leading 1
    uint32_t mantissa_b = get_mantissa(b) | 0x80;

    // Shift mantissa_a to increase precision before division
    uint32_t mantissa_numerator = mantissa_a << 8; // Shift by 8 bits for more precision
    uint32_t result_mantissa = mantissa_numerator / mantissa_b;

    int32_t result_exp = exp_a - exp_b + 127;
    uint32_t result_sign = sign_a ^ sign_b;

    // Handle case where division result is zero
    if (result_mantissa == 0) {
        return assemble_bf16(0, 0, 0);
    }

    // Normalize result mantissa
    while (result_mantissa >= 0x100) { // If mantissa >= 256
        result_mantissa >>= 1;
        result_exp++;
    }
    while (result_mantissa < 0x80 && result_exp > 0) { // If mantissa < 128
        result_mantissa <<= 1;
        result_exp--;
    }

    // Handle exponent overflow/underflow
    if (result_exp >= 0xFF) {
        // Exponent overflow, return infinity
        return assemble_bf16(result_sign, 0xFF, 0);
    } else if (result_exp <= 0) {
        // Exponent underflow, return zero
        return assemble_bf16(0, 0, 0);
    }

    // Round the mantissa (optional, for better accuracy)
    // Implement rounding to nearest even if desired

    // Remove the implicit leading 1
    uint32_t mantissa = result_mantissa & 0x7F;

    return assemble_bf16(result_sign, result_exp, mantissa);
}

