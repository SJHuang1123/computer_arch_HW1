.data
test: .word 0x40000000, 0x40400000, 0x40800000, 0x40c00000 
str1: .asciz "Matrix:\n"   
str3: .asciz "\nDeterminant:\n"  
newline: .asciz "\n" 
                  
.text
.globl main
                    
main:    
    # Print "Matrix:\n"
    la s0, test
    la a0, str1          # Load string address
    li a7, 4             # System call code 4: Print string
    ecall
    li t1, 2
    li, t0, 0
print_matrix:
    bge t0, t1, print_result  # If i >= 2, check for singularity
    li t2, 0                       # Inner loop index j initialized to 0

print_inner_loop:
    bge t2, t1, next_row           # If j >= 2, move to the next row
    lw  a0, 0(s0)                   # Load the current matrix element into a0
    li a7, 2                       # System call code 2: Print float
    ecall
    addi s0, s0, 4                 # Move to the next element
    li a0, 32                      # Print space
    li a7, 11                      # System call code 11: Print character
    ecall
    addi t2, t2, 1                 # Increment j
    j print_inner_loop             # Repeat inner loop

next_row:
    la a0, newline                 # Print newline
    li a7, 4                       # System call code 4: Print string
    ecall
    addi t0, t0, 1                 # Increment i
    j print_matrix                 # Repeat outer loop                 
print_result:
    la t1, test
    lw a0, 0(t1)
    call fp32_to_bf16
    sh a0, 0(t1)
    lw a0, 4(t1)
    call fp32_to_bf16
    sh a0, 2(t1)
    lw a0, 8(t1)
    call fp32_to_bf16
    sh a0, 4(t1)
    lw a0, 12(t1)
    call fp32_to_bf16
    sh a0, 6(t1)
    mv a0, t1
    call determinant
    mv t0, a0
    
    la a0, str3          # Load string address
    li a7, 4             # System call code 4: Print string
    ecall
    mv a0, t0
    slli a0, a0, 16
    li a7, 2             # System call code 4: Print string
    ecall
end_program:
    li a7, 10                      # End the program
    ecall

bf16_abs:             
        li      t0, 0x7FFF        # Load immediate mask (32768 - 1) directly
        and     a0, a0, t0        # Apply mask to zero out the sign bit
        jr      ra                # Return to caller
assemble_bf16:
        andi    a0, a0, 1          # Mask sign bit (only LSB is used)
        slli    a0, a0, 15         # Move sign bit to its position
        andi    a1, a1, 255        # Mask exponent (8 bits)
        slli    a1, a1, 7          # Shift exponent to its position
        andi    a2, a2, 127        # Mask mantissa (7 bits)
        or      a0, a0, a1         # Combine sign and exponent
        or      a0, a0, a2         # Combine result with mantissa
        jr      ra                 # Return to caller
bf16_add:
    # Prologue
    addi    sp, sp, -40          # Allocate stack space
    sw      ra, 36(sp)           # Save return address
    sw      s0, 32(sp)           # Save s0
    sw      s1, 28(sp)           # Save s1
    sw      s2, 24(sp)           # Save s2
    sw      s3, 20(sp)           # Save s3
    sw      s4, 16(sp)           # Save s4
    sw      s5, 12(sp)           # Save s5
    sw      s6, 8(sp)            # Save s6
    sw      s7, 4(sp)            # Save s7

    # Extract sign, exponent, and mantissa for operand a
    srli    s0, a0, 15           # s0 = sign_a
    srli    s2, a0, 7            # s2 = exp_a (includes sign bit)
    andi    s2, s2, 0xFF         # Mask to get 8-bit exponent
    andi    s4, a0, 0x7F         # s4 = mantissa_a
    ori     s4, s4, 0x80         # Add implicit leading 1 to mantissa_a
 
    # Extract sign, exponent, and mantissa for operand b
    srli    s1, a1, 15           # s1 = sign_b
    srli    s3, a1, 7            # s3 = exp_b
    andi    s3, s3, 0xFF         # Mask to get 8-bit exponent
    andi    s5, a1, 0x7F         # s5 = mantissa_b
    ori     s5, s5, 0x80         # Add implicit leading 1 to mantissa_b

    # Align exponents
    blt     s2, s3, align_exp_b
align_exp_a:
    sub     t0, s2, s3           # t0 = exp_a - exp_b
    mv      t1, s5               # t1 = mantissa_b
align_loop_a:
    beq     t0, zero, aligned_a
    srli    t1, t1, 1            # Shift mantissa_b right
    addi    t0, t0, -1           # Decrement t0
    j       align_loop_a
aligned_a:
    mv      s5, t1               # Update mantissa_b
    j       perform_add_sub
align_exp_b:
    sub     t0, s3, s2           # t0 = exp_b - exp_a
    mv      t1, s4               # t1 = mantissa_a
align_loop_b:
    beq     t0, zero, aligned_b
    srli    t1, t1, 1            # Shift mantissa_a right
    addi    t0, t0, -1           # Decrement t0
    j       align_loop_b
aligned_b:
    mv      s4, t1               # Update mantissa_a
    mv      s2, s3               # Set exp_a = exp_b

perform_add_sub:
    # Perform addition or subtraction
    xor     t0, s0, s1           # t0 = sign_a ^ sign_b
    beq     t0, zero, signs_equal
    # Signs are different
    blt     s4, s5, mant_b_ge_a
    sub     s6, s4, s5           # s6 = mantissa_a - mantissa_b
    mv      s7, s0               # s7 = sign_a
    j       handle_zero_result
mant_b_ge_a:
    sub     s6, s5, s4           # s6 = mantissa_b - mantissa_a
    mv      s7, s1               # s7 = sign_b
    j       handle_zero_result
signs_equal:
    # Signs are the same
    add     s6, s4, s5           # s6 = mantissa_a + mantissa_b
    mv      s7, s0               # s7 = sign_a

handle_zero_result:
    # Check for zero result
    beq     s6, zero, return_zero

normalize_result:
    # Normalize result mantissa
    li      t0, 0xFF             # t0 = 255 (max mantissa before normalization)
    bgt     s6, t0, normalize_overflow
    li      t0, 0x80             # t0 = 128 (implicit leading 1)
    blt     s6, t0, normalize_underflow
    j       check_overflow       # Mantissa is normalized

normalize_overflow:
    srli    s6, s6, 1            # Shift mantissa right
    addi    s2, s2, 1            # Increment exponent
    bge     s2, t0, check_overflow  # Check for exponent overflow
    li      t1, 0xFF
    bgt     s6, t1, normalize_overflow
    j       check_overflow

normalize_underflow:
    ble     s2, zero, check_overflow  # Check for exponent underflow
    slli    s6, s6, 1            # Shift mantissa left
    addi    s2, s2, -1           # Decrement exponent
    blt     s6, t0, normalize_underflow
    j       check_overflow

check_overflow:
    # Handle exponent overflow/underflow
    li      t0, 0xFF
    bge     s2, t0, return_infinity   # Exponent overflow
    ble     s2, zero, return_zero     # Exponent underflow
    j       assemble_result

return_zero:
    # Return zero
    li      a0, 0                # sign = 0
    li      a1, 0                # exponent = 0
    li      a2, 0                # mantissa = 0
    jal     ra, assemble_bf16    # Return zero
    j       epilogue_add

return_infinity:
    # Return infinity
    mv      a0, s7               # result_sign
    li      a1, 0xFF             # exponent = 0xFF
    li      a2, 0                # mantissa = 0
    jal     ra, assemble_bf16    # Return infinity
    j       epilogue_add

assemble_result:
    # Remove the implicit leading 1
    andi    s6, s6, 0x7F         # Clear the leading 1 bit

    # Prepare arguments for assemble_bf16
    mv      a0, s7               # result_sign
    mv      a1, s2               # exponent
    mv      a2, s6               # mantissa
    jal     ra, assemble_bf16    # Assemble and return bf16
    j       epilogue_add

epilogue_add:
    # Epilogue
    lw      ra, 36(sp)           # Restore return address
    lw      s0, 32(sp)
    lw      s1, 28(sp)
    lw      s2, 24(sp)
    lw      s3, 20(sp)
    lw      s4, 16(sp)
    lw      s5, 12(sp)
    lw      s6, 8(sp)
    lw      s7, 4(sp)
    addi    sp, sp, 40
    jr      ra

# Function to multiply two bfloat16 numbers
# bf16_t bf16_mul(bf16_t a, bf16_t b)
bf16_mul:
    # Prologue: Save registers and allocate stack space
    addi    sp, sp, -40          # Allocate stack space
    sw      ra, 36(sp)           # Save return address
    sw      s0, 32(sp)           # Save s0
    sw      s1, 28(sp)           # Save s1
    sw      s2, 24(sp)           # Save s2
    sw      s3, 20(sp)           # Save s3
    sw      s4, 16(sp)           # Save s4
    sw      s5, 12(sp)           # Save s5
    sw      s6, 8(sp)            # Save s6
    sw      s7, 4(sp)            # Save s7

    # Extract sign, exponent, and mantissa for 'a'
    srli    s0, a0, 15           # s0 = a >> 15 (sign_a)
    andi    s0, s0, 0x1          # s0 = sign_a & 0x1

    srli    s2, a0, 7            # s2 = a >> 7
    andi    s2, s2, 0xFF         # s2 = exp_a

    andi    s4, a0, 0x7F         # s4 = mantissa_a
    ori     s4, s4, 0x80         # s4 |= 0x80 (restore implicit leading 1)

    # Extract sign, exponent, and mantissa for 'b'
    srli    s1, a1, 15           # s1 = b >> 15 (sign_b)
    andi    s1, s1, 0x1          # s1 = sign_b & 0x1

    srli    s3, a1, 7            # s3 = b >> 7
    andi    s3, s3, 0xFF         # s3 = exp_b

    andi    s5, a1, 0x7F         # s5 = mantissa_b
    ori     s5, s5, 0x80         # s5 |= 0x80 (restore implicit leading 1)

    # Compute result_sign = sign_a ^ sign_b
    xor     s6, s0, s1           # s6 = result_sign

    # Compute result_exp = exp_a + exp_b - 127
    add     s7, s2, s3           # s7 = exp_a + exp_b
    addi    s7, s7, -127         # s7 = result_exp

    # Multiply mantissas: mantissa_product = mantissa_a * mantissa_b
    mul     t0, s4, s5           # t0 = mantissa_product (16 bits)

    # Normalization
    # Check if bit 15 of result_mantissa is set
    li      t1, 0x8000           # t1 = 0x8000
    and     t1, t0, t1           # t1 = t0 & 0x8000
    bnez    t1, shift_right_by_one

    # Check if bit 14 is set
    li      t1, 0x4000           # t1 = 0x4000
    and     t1, t0, t1           # t1 = t0 & 0x4000
    bnez    t1, mantissa_normalized

    # Neither bit 15 nor bit 14 is set, shift left to normalize
normalize_shift_left:
    li      t1, 0x4000           # t1 = 0x4000
normalize_shift_left_loop:
    and     t1, t0, t1           # t1 = t0 & 0x4000
    bnez    t1, mantissa_normalized

    slli    t0, t0, 1            # t0 <<= 1
    addi    s7, s7, -1           # result_exp--

    blez    s7, underflow        # Check for exponent underflow
    j       normalize_shift_left_loop

shift_right_by_one:
    srli    t0, t0, 1            # t0 >>= 1
    addi    s7, s7, 1            # result_exp++

    li      t1, 0xFF
    bge     s7, t1, overflow     # Check for exponent overflow

    j       mantissa_normalized

mantissa_normalized:
    # Adjust mantissa to 7 bits
    srli    t2, t0, 7            # t2 = t0 >> 7
    # Remove implicit leading 1
    andi    t2, t2, 0x7F         # t2 = t2 & 0x7F

    # Check for exponent overflow and underflow
check_overflow_underflow:
    li      t1, 0xFF
    bge     s7, t1, overflow     # Exponent overflow
    blez    s7, underflow        # Exponent underflow

    # Exponent is within valid range
    # Assemble the bf16 result
    slli    s6, s6, 15           # s6 = result_sign << 15
    slli    s7, s7, 7            # s7 = result_exp << 7
    or      a0, s6, s7           # a0 = result_sign | result_exp
    or      a0, a0, t2           # a0 |= mantissa
    j       epilogue_mul

overflow:
    # Return infinity
    slli    s6, s6, 15           # s6 = result_sign << 15
    li      s7, 0xFF             # s7 = exponent = 0xFF
    slli    s7, s7, 7            # s7 = exponent << 7
    or      a0, s6, s7           # a0 = result_sign | exponent
    # Mantissa is zero
    j       epilogue_mul

underflow:
    # Return zero
    li      a0, 0                # a0 = 0
    j       epilogue_mul

epilogue_mul:
    # Epilogue: Restore registers and return
    lw      ra, 36(sp)           # Restore return address
    lw      s0, 32(sp)
    lw      s1, 28(sp)
    lw      s2, 24(sp)
    lw      s3, 20(sp)
    lw      s4, 16(sp)
    lw      s5, 12(sp)
    lw      s6, 8(sp)
    lw      s7, 4(sp)
    addi    sp, sp, 40
    jr      ra

determinant:
    # Prologue
    addi    sp, sp, -24          # Allocate stack space
    sw      ra, 20(sp)           # Save return address
    sw      s0, 16(sp)           # Save s0
    sw      s1, 12(sp)           # Save s1
    sw      s2, 8(sp)            # Save s2

    # Load matrix elements
    mv      s0, a0
    lh      a0, 0(s0)
    lh      a1, 6(s0)
    call    bf16_mul
    mv      s1, a0 
    lh      a0, 2(s0)
    lh      a1, 4(s0)
    call    bf16_mul
    mv      a1, a0
    mv      a0, s1
    li      t0, 0x8000   
    xor     a1, a1, t0
    jal     ra, bf16_add  

    # Epilogue
    lw      s2, 8(sp)            # Restore s2
    lw      s1, 12(sp)           # Restore s1
    lw      s0, 16(sp)           # Restore s0
    lw      ra, 20(sp)           # Restore return address
    addi    sp, sp, 24           # Deallocate stack space
    jr      ra                   # Return to caller
fp32_to_bf16:
    li t0, 0x7fffffff
    and t0, t0, a0 # t0 = abs(f)
    li t1, 0x7f800000 # check whether number is NaN 
    ble t0, t1, normal # if t0 <= t1 then target
    srli t0, a0, 16
    ori t0, t0, 64
    j  ret
normal:
    srli t0, a0, 0x10
    andi t0, t0, 1
    li t1, 0x7fff
    add t0, t0, t1
    add t0, a0, t0
    srli t0, t0, 0x10
ret:
    mv a0, t0
    jr ra
