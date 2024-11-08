import math
import numpy as np
import time
import mpmath
import numpy as np
from math import acos, degrees, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import csv

start_time = time.time()

class BlackHole:
    
    def __init__(self, spin, r_0, theta_0, phi_0, alpha_0, beta_0):
        self.spin = spin
        self.r_0 = r_0
        self.theta_0 = theta_0
        self.phi_0 = phi_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mass = 1.0


mpmath.mp.dps = 200
M = 1.0
spin = mpmath.mpf("0.999999999999999999999999999999999999999999999999999999999999999999999999999999999")
step_number = 1000000

# Function for addition
def plus_ultra(x, y, prec):
    mpmath.mp.dps = prec  # Set decimal precision
    return mpmath.mpf(x) + mpmath.mpf(y)
# Function for subtraction
def minus_ultra(x, y, prec):
    mpmath.mp.dps = prec
    return mpmath.mpf(x) - mpmath.mpf(y)
# Function for multiplication
def times_ultra(x, y, prec):
    mpmath.mp.dps = prec
    return mpmath.mpf(x) * mpmath.mpf(y)
# Function for division
def divide_ultra(x, y, prec):
    mpmath.mp.dps = prec
    return mpmath.mpf(x) / mpmath.mpf(y)
# Function for power (x^y)
def power_ultra(x, y, prec):
    mpmath.mp.dps = prec
    return mpmath.power(mpmath.mpf(x), mpmath.mpf(y))
# Function for square root
def sqrt_ultra(x, prec):
    mpmath.mp.dps = prec
    return mpmath.sqrt(mpmath.mpf(x))
# Function for sine
def sin_ultra(x, prec):
    mpmath.mp.dps = prec
    return mpmath.sin(mpmath.mpf(x))
# Function for cosine
def cos_ultra(x, prec):
    mpmath.mp.dps = prec
    return mpmath.cos(mpmath.mpf(x))

def dot_ultra(vec1, vec2, prec):
    # Ensure the precision is set
    mpmath.mp.dps = prec
    
    # Ensure both vectors are of the same length
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same length")
    
    # Calculate the dot product as the sum of the products of corresponding elements
    dot_product = mpmath.mpf(0)  # Initialize to 0 with high precision
    for i in range(len(vec1)):
        dot_product += mpmath.mpf(vec1[i]) * mpmath.mpf(vec2[i])
    
    return dot_product

def C_010_ultra(M, spin, r, theta, prec):
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate ultra-precision values
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    cos_theta = cos_ultra(theta, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)

    # Numerator (same as before)
    numerator = times_ultra(
        M,
        times_ultra(
            plus_ultra(r_squared, spin_squared, prec),
            minus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec),
            prec
        ),
        prec
    )

    # Intermediate steps in the denominator (breaking down term2_final)
    term1 = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    term1_squared = times_ultra(term1, term1, prec)

    term2_inner = times_ultra(mpmath.mpf(2.0), times_ultra(M, r, prec), prec)
    term2 = minus_ultra(r_squared, term2_inner, prec)
    term2_final = plus_ultra(term2, spin_squared, prec)  # Adjusted to addition for clarity

    # Printing intermediate values specifically for term2_final
    #print(f"Ultra-precision function - r_squared: {r_squared}")
    #print(f"Ultra-precision function - spin_squared: {spin_squared}")
    #print(f"Ultra-precision function - term2_inner: {term2_inner}")
    #print(f"Ultra-precision function - term2: {term2}")
    #print(f"Ultra-precision function - term2_final: {term2_final}")  # Compare this closely with the original

    # Full denominator calculation
    denominator = times_ultra(term1_squared, term2_final, prec)
    #print(f"Ultra-precision function - Denominator: {denominator}")

    # Final result
    return divide_ultra(numerator, denominator, prec)

def C_002_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec
    
    # Convert inputs to mpmath.mpf for consistent ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values with ultra functions
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)

    # Numerator: -2 * M * spin^2 * r * sin(theta) * cos(theta)
    numerator = times_ultra(
        mpmath.mpf(-2.0),
        times_ultra(
            M,
            times_ultra(
                spin_squared,
                times_ultra(
                    r,
                    times_ultra(sin_theta, cos_theta, prec),
                    prec
                ),
                prec
            ),
            prec
        ),
        prec
    )
    
    # Intermediate term for the denominator: (r^2 + spin^2 * cos(theta)^2)
    term1 = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    term1_squared = times_ultra(term1, term1, prec)  # Square term1 to complete the denominator

    # Full denominator
    denominator = term1_squared
    
    # Final result
    return divide_ultra(numerator, denominator, prec)

def C_013_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec
    
    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute squared values
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    sin_theta_squared = power_ultra(sin_theta, 2, prec)
    cos_theta = cos_ultra(theta, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)

    # Step 2: Calculate the terms in the numerator
    # Inner term for numerator: (spin^2 * cos(theta)^2 * (spin^2 - r^2))
    inner_term1 = times_ultra(spin_squared, cos_theta_squared, prec)
    inner_term2 = minus_ultra(spin_squared, r_squared, prec)
    term1 = times_ultra(inner_term1, inner_term2, prec)

    # Outer term for numerator: (r^2 * (spin^2 + 3 * r^2))
    inner_term3 = plus_ultra(spin_squared, times_ultra(3.0, r_squared, prec), prec)
    term2 = times_ultra(r_squared, inner_term3, prec)

    # Complete numerator calculation
    numerator = times_ultra(
        mpmath.mpf(2.0),
        times_ultra(
            M,
            times_ultra(
                spin,
                sin_theta_squared,
                prec
            ),
            prec
        ),
        prec
    )
    numerator = times_ultra(numerator, minus_ultra(term1, term2, prec), prec)

    # Step 3: Denominator calculation
    # Intermediate term1 for denominator: (r^2 + spin^2 * cos(theta)^2)
    term3 = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    term3_squared = times_ultra(term3, term3, prec)

    # Intermediate term2 for denominator: (r^2 - 2 * M * r + spin^2)
    term4 = minus_ultra(r_squared, times_ultra(2.0, times_ultra(M, r, prec), prec), prec)
    term5 = plus_ultra(term4, spin_squared, prec)

    # Full denominator
    denominator = times_ultra(mpmath.mpf(2.0), times_ultra(term3_squared, term5, prec), prec)

    # Final result
    return divide_ultra(numerator, denominator, prec)

def C_023_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec
    
    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    sin_theta_squared = power_ultra(sin_theta, 2, prec)
    sin_theta_cubed = times_ultra(sin_theta_squared, sin_theta, prec)  # sin^3(theta)
    cos_theta = cos_ultra(theta, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)

    # Step 2: Calculate the numerator: 2 * M * spin^3 * r * sin^3(theta) * cos(theta)
    numerator = times_ultra(
        mpmath.mpf(2.0),
        times_ultra(
            M,
            times_ultra(
                spin,
                times_ultra(
                    spin_squared,
                    times_ultra(
                        r,
                        times_ultra(sin_theta_cubed, cos_theta, prec),
                        prec
                    ),
                    prec
                ),
                prec
            ),
            prec
        ),
        prec
    )

    # Step 3: Calculate the denominator: (r^2 + spin^2 * cos^2(theta))^2
    denominator = power_ultra(
        plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec),
        2,
        prec
    )

    # Final result
    return divide_ultra(numerator, denominator, prec)

def C_111_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec
    
    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    sin_theta_squared = power_ultra(sin_theta, 2, prec)
    cos_theta = cos_ultra(theta, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)

    # Step 2: Calculate the numerator
    term1 = times_ultra(mpmath.mpf(2.0), r, prec)
    term1 = times_ultra(term1, spin_squared, prec)
    term1 = times_ultra(term1, sin_theta_squared, prec)

    term2 = times_ultra(mpmath.mpf(2.0), M, prec)
    term2 = times_ultra(term2, minus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec), prec)

    numerator = minus_ultra(term1, term2, prec)

    # Step 3: Calculate the denominator
    denominator_term1 = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    denominator_term2 = minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec)
    denominator_term2 = plus_ultra(denominator_term2, spin_squared, prec)

    denominator = times_ultra(mpmath.mpf(2.0), denominator_term1, prec)
    denominator = times_ultra(denominator, denominator_term2, prec)

    # Final result
    return divide_ultra(numerator, denominator, prec)

def C_100_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec
    
    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values
    r_squared = power_ultra(r, 2, prec)
    #print(f"Ultra r_squared: {r_squared}")

    spin_squared = power_ultra(spin, 2, prec)
    #print(f"Ultra spin_squared: {spin_squared}")

    cos_theta_squared = power_ultra(cos_ultra(theta, prec), 2, prec)
    #print(f"Ultra cos_theta_squared: {cos_theta_squared}")

    # Step 2: Calculate the numerator
    term1 = times_ultra(mpmath.mpf(2.0), M, prec)  # 2M
    #print(f"Ultra 2M: {term1}")
    
    r2 = times_ultra(mpmath.mpf(2.0), r, prec)

    term1 = times_ultra(term1, (plus_ultra((minus_ultra(r_squared, r2, prec)), spin_squared, prec)), prec)

    term1 = times_ultra(term1, minus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec), prec)
    #print(f"Ultra Numerator after second calculation: {term1}")

    # Step 3: Calculate the denominator
    denominator_term = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    #print(f"Ultra Denominator term: {denominator_term}")

    denominator = times_ultra(denominator_term, denominator_term, prec)  # Squared term
    #print(f"Ultra Denominator (squared): {denominator}")

    denominator = times_ultra(denominator, denominator_term, prec)  # Multiply again by the denominator term
    #print(f"Ultra Denominator (cubed): {denominator}")

    # Final result
    result = divide_ultra(term1, times_ultra(mpmath.mpf(2.0), denominator, prec), prec)
    #print(f"Ultra Final result: {result}")

    return result

def C_122_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values
    r_squared = power_ultra(r, 2, prec)
    #print(f"Ultra r_squared: {r_squared}")

    spin_squared = power_ultra(spin, 2, prec)
    #print(f"Ultra spin_squared: {spin_squared}")

    cos_theta_squared = power_ultra(cos_ultra(theta, prec), 2, prec)
    #print(f"Ultra cos_theta_squared: {cos_theta_squared}")

    # Step 2: Calculate the numerator
    # (r * (r^2 - 2 * M * r + spin^2))
    term1 = minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec)
    #print(f"Ultra Numerator after first calculation: {term1}")

    term1 = plus_ultra(term1, spin_squared, prec)
    #print(f"Ultra Numerator after adding spin_squared: {term1}")

    numerator = times_ultra(r, term1, prec)
    #print(f"Ultra Final Numerator: {numerator}")

    # Step 3: Calculate the denominator
    # (r^2 + spin^2 * cos^2(theta))
    denominator = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    #print(f"Ultra Denominator: {denominator}")

    # Final result with negation
    result = -divide_ultra(numerator, denominator, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_133_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta_squared = power_ultra(sin_ultra(theta, prec), 2, prec)
    cos_theta_squared = power_ultra(cos_ultra(theta, prec), 2, prec)

    # Step 2: Calculate the main numerator
    # (sin(theta)^2 * (r^2 - 2 * M * r + spin^2))
    term1 = minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec)
    term1 = plus_ultra(term1, spin_squared, prec)
    numerator_main = times_ultra(sin_theta_squared, term1, prec)
    #print(f"Ultra Numerator Main: {numerator_main}")

    # Step 3: Calculate the main denominator
    # 2 * (r^2 + spin^2 * cos^2(theta))^3
    inner_denominator = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    denominator_main = times_ultra(mpmath.mpf(2.0), power_ultra(inner_denominator, 3, prec), prec)
    #print(f"Ultra Denominator Main: {denominator_main}")

    # Step 4: Calculate the nested term in parentheses
    # -2 * r * (inner_denominator^2) + 2 * M * spin^2 * sin(theta)^2 * (r^2 - spin^2 * cos(theta)^2)
    term2_1_one = times_ultra(mpmath.mpf(2.0), r, prec)
    term2_1 = -times_ultra(term2_1_one, power_ultra(inner_denominator, 2, prec), prec)
    #print(f"Ultra Term 2 Part 1: {term2_1}")

    term2_2 = times_ultra(mpmath.mpf(2.0), spin_squared, prec)
    term2_2 = times_ultra(term2_2, sin_theta_squared, prec)
    inner_term = minus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    term2_2 = times_ultra(term2_2, inner_term, prec)
    #print(f"Ultra Term 2 Part 2: {term2_2}")

    # Final nested term (sum of term2_1 and term2_2)
    nested_term = plus_ultra(term2_1, term2_2, prec)
    #print(f"Ultra Nested Term: {nested_term}")

    # Step 5: Complete numerator by multiplying with nested_term
    numerator_final = times_ultra(numerator_main, nested_term, prec)
    #print(f"Ultra Final Numerator: {numerator_final}")

    # Final result
    result = divide_ultra(numerator_final, denominator_main, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_103_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Compute intermediate values
    r_squared = power_ultra(r, 2, prec)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta_squared = power_ultra(sin_ultra(theta, prec), 2, prec)
    cos_theta_squared = power_ultra(cos_ultra(theta, prec), 2, prec)

    # Step 2: Calculate the main numerator
    # (2 * M * (r^2 - 2 * M * r + spin^2) * spin * sin(theta)^2 * (r^2 - spin^2 * cos(theta)^2))
    term1 = minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec)
    term1 = plus_ultra(term1, spin_squared, prec)
    term1 = times_ultra(times_ultra(mpmath.mpf(2.0), term1, prec), spin, prec)
    term1 = times_ultra(term1, sin_theta_squared, prec)
    #print(f"Ultra Numerator First Part: {term1}")

    # (r^2 - spin^2 * cos(theta)^2)
    term2 = minus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    numerator_final = times_ultra(term1, term2, prec)
    #print(f"Ultra Final Numerator: {numerator_final}")

    # Step 3: Calculate the main denominator
    # 2 * (r^2 + spin^2 * cos^2(theta))^3
    inner_denominator = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    denominator_main = times_ultra(mpmath.mpf(2.0), power_ultra(inner_denominator, 3, prec), prec)
    #print(f"Ultra Denominator Main: {denominator_main}")

    # Step 4: Final result (with negative sign)
    result = -divide_ultra(numerator_final, denominator_main, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_112_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Calculate spin^2 * sin(theta) * cos(theta)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)
    numerator = times_ultra(times_ultra(spin_squared, sin_theta, prec), cos_theta, prec)
    #print(f"Ultra Numerator: {numerator}")

    # Step 2: Calculate the denominator (r^2 + spin^2 * cos(theta)^2)
    r_squared = power_ultra(r, 2, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)
    denominator = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    #print(f"Ultra Denominator: {denominator}")

    # Step 3: Final result (with negative sign)
    result = -divide_ultra(numerator, denominator, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_222_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Calculate spin^2 * sin(theta) * cos(theta) for the numerator
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)
    numerator = times_ultra(times_ultra(spin_squared, sin_theta, prec), cos_theta, prec)
    #print(f"Ultra Numerator: {numerator}")

    # Step 2: Calculate the denominator (r^2 + spin^2 * cos(theta)^2)
    r_squared = power_ultra(r, 2, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)
    denominator = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    #print(f"Ultra Denominator: {denominator}")

    # Step 3: Final result (with negative sign)
    result = -divide_ultra(numerator, denominator, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_200_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Calculate 2 * M * spin^2 * r * sin(theta) * cos(theta) for the numerator
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)
    term1 = times_ultra(mpmath.mpf(2.0), M, prec)
    term1 = times_ultra(term1, spin_squared, prec)
    term1 = times_ultra(term1, r, prec)
    numerator = times_ultra(term1, times_ultra(sin_theta, cos_theta, prec), prec)
    #print(f"Ultra Numerator: {numerator}")

    # Step 2: Calculate the denominator (r^2 + spin^2 * cos(theta)^2)^3
    r_squared = power_ultra(r, 2, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)
    inner_term = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    denominator = power_ultra(inner_term, 3, prec)
    #print(f"Ultra Denominator: {denominator}")

    # Step 3: Final result (with negative sign)
    result = -divide_ultra(numerator, denominator, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_211_ultra(M, spin, r, theta, prec):
    # Set precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for ultra-precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Step 1: Calculate the numerator: spin^2 * sin(theta) * cos(theta)
    spin_squared = power_ultra(spin, 2, prec)
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)
    numerator = times_ultra(spin_squared, times_ultra(sin_theta, cos_theta, prec), prec)
    #print(f"Ultra Numerator: {numerator}")

    # Step 2: Calculate the first term of the denominator: r^2 + spin^2 * cos(theta)^2
    r_squared = power_ultra(r, 2, prec)
    cos_theta_squared = power_ultra(cos_theta, 2, prec)
    term1 = plus_ultra(r_squared, times_ultra(spin_squared, cos_theta_squared, prec), prec)
    #print(f"Ultra Denominator First Term: {term1}")

    # Step 3: Calculate the second term of the denominator: r^2 - 2*M*r + spin^2
    term2 = minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), times_ultra(M, r, prec), prec), prec)
    term2 = plus_ultra(term2, spin_squared, prec)
    #print(f"Ultra Denominator Second Term: {term2}")

    # Step 4: Combine both terms for the denominator
    denominator = times_ultra(term1, term2, prec)
    #print(f"Ultra Denominator: {denominator}")

    # Step 5: Final result
    result = divide_ultra(numerator, denominator, prec)
    #print(f"Ultra Final Result: {result}")

    return result

def C_233_ultra(M, spin, r, theta, prec):
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sin_cos = times_ultra(sin_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    delta = plus_ultra(spin_squared, minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec), prec)
    a_1 = power_ultra(plus_ultra(r_squared, spin_squared, prec), 2, prec)
    a_2 = times_ultra(spin_squared, times_ultra(delta, sin_squared, prec), prec)
    aa = minus_ultra(a_1, a_2, prec)
    
    power_3 = power_ultra(sigma, 3, prec)
    frac = divide_ultra(sin_cos, power_3, prec)
    result_1 = times_ultra(mpmath.mpf(-1.0), frac, prec)
    
    sub = times_ultra(plus_ultra(r_squared, spin_squared, prec), times_ultra(mpmath.mpf(2.0), times_ultra(spin_squared, times_ultra(r, sin_squared, prec), prec), prec), prec)
    result_2 = plus_ultra(times_ultra(aa, sigma, prec), sub, prec)
    
    result = times_ultra(result_1, result_2, prec)
    
    return result

def C_203_ultra(M, spin, r, theta, prec):
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec) 
    power_3 = power_ultra(sigma, 3, prec)
    sin_cos = times_ultra(sin_theta, cos_theta, prec)
    
    numerator = times_ultra(mpmath.mpf(2.0), times_ultra(spin, times_ultra(r, times_ultra(plus_ultra(r_squared, spin_squared, prec), sin_cos, prec), prec), prec), prec)
    
    result = divide_ultra(numerator, power_3, prec)
    
    return result

def C_212_ultra(M,  spin,  r,  theta, prec):
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    
    result = divide_ultra(r, sigma, prec)
    
    return result

def C_301_ultra(M,  spin,  r,  theta, prec):
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    power_2 = power_ultra(sigma, 2, prec)
    delta = plus_ultra(spin_squared, minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec), prec)
    
    parenthesis = minus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    numerator = times_ultra(mpmath.mpf(2.0), times_ultra(spin, parenthesis, prec), prec)
    denominator = times_ultra(mpmath.mpf(2.0), times_ultra(power_2, delta, prec), prec)
    
    result = divide_ultra(numerator, denominator, prec)
    
    return result

def C_302_ultra(M,  spin,  r,  theta, prec):
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    power_2 = power_ultra(sigma, 2, prec)
    delta = plus_ultra(spin_squared, minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec), prec)
    cot = divide_ultra(cos_theta, sin_theta, prec)
    
    numerator = times_ultra(mpmath.mpf(2.0), times_ultra(spin, times_ultra(r, cot, prec), prec), prec)
    fraction = divide_ultra(numerator, power_2, prec)
    result = times_ultra(mpmath.mpf(-1.0), fraction, prec)
    
    return result

def C_313_ultra(M,  spin,  r,  theta, prec):
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    power_2 = power_ultra(sigma, 2, prec)
    delta = plus_ultra(spin_squared, minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec), prec)
    cot = divide_ultra(cos_theta, sin_theta, prec)
    spin_4 = power_ultra(spin, 4, prec)
    
    a = times_ultra(spin_4, times_ultra(sin_squared, cos_squared, prec), prec)
    parenthesis = plus_ultra(sigma, plus_ultra(r_squared, spin_squared, prec), prec)
    b = times_ultra(r_squared, parenthesis, prec) 
    
    subsub = (minus_ultra(a, b, prec))
    sub = times_ultra(mpmath.mpf(2.0), subsub, prec)
    numerator = plus_ultra(times_ultra(mpmath.mpf(2.0), times_ultra(power_2, r, prec), prec), sub, prec)
    
    denominator = times_ultra(mpmath.mpf(2.0), times_ultra(power_2, delta, prec), prec)
    
    result = divide_ultra(numerator, denominator, prec)
    
    return result

def C_323_ultra(M,  spin,  r,  theta, prec):
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    power_2 = power_ultra(sigma, 2, prec)
    delta = plus_ultra(spin_squared, minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec), prec)
    cot = divide_ultra(cos_theta, sin_theta, prec)


    result1 = divide_ultra(cot, power_2, prec)
    sub = times_ultra(mpmath.mpf(2.0), times_ultra(spin_squared, times_ultra(r, sin_squared, prec), prec), prec)
    result2 = plus_ultra(power_2, sub, prec)

    result = times_ultra(result1, result2, prec)
    
    return result

#########################################################################################################################################
# CHRISTOFFEL SYMBOLS

def fd2tds2( M,  r,  theta,  dtds,  drds,  dthds,  dphids, prec):
    return -1.0*(C_010_ultra(M, spin, r, theta, prec)) * drds * dtds * 2.0 -1.0*(C_002_ultra(M, spin, r, theta, prec)) * dtds * dthds * 2.0 -1.0*(C_013_ultra(M, spin, r, theta, prec)) * drds * dphids * 2.0 -1.0*(C_023_ultra(M, spin, r, theta, prec)) * dthds * dphids * 2.0

def fd2rds2( M,  r,  theta,  dtds,  drds,  dthds,  dphids, prec):
    return -1.0*C_111_ultra(M, spin, r, theta, prec) * drds * drds -1.0*C_100_ultra(M, spin, r, theta, prec) * dtds * dtds -1.0*C_122_ultra(M, spin, r, theta, prec) * dthds *dthds -1.0*C_133_ultra(M, spin, r, theta, prec) * dphids * dphids -1.0*C_103_ultra(M, spin, r, theta, prec) * dtds * dphids * 2.0 -1.0*C_112_ultra(M, spin, r, theta, prec) * drds * dthds * 2.0

def fd2thds2( M,  r,  theta,  dtds,  drds,  dthds,  dphids, prec):
    return -1.0*C_222_ultra(M, spin, r, theta, prec) * dthds * dthds -1.0*C_200_ultra(M, spin, r, theta, prec) * dtds * dtds -1.0*C_211_ultra(M, spin, r, theta, prec) * drds * drds -1.0*C_233_ultra(M, spin, r, theta, prec) * dphids * dphids -1.0*C_203_ultra(M, spin, r, theta, prec) * dtds * dphids * 2.0 -1.0*C_212_ultra(M, spin, r, theta, prec) * drds * dthds * 2.0

def fd2phids2( M,  r,  theta,  dtds,  drds,  dthds,  dphids, prec):
    return -1.0*C_301_ultra(M, spin, r, theta, prec) * dtds * drds * 2.0 -1.0*C_302_ultra(M, spin, r, theta, prec) * dtds * dthds * 2.0 -1.0*C_313_ultra(M, spin, r, theta, prec) * drds * dphids * 2.0 -1.0*C_323_ultra(M, spin, r, theta, prec) * dthds * dphids * 2.0 

#########################################################################################################################################################

def Correction_drds_ultra(r, M, theta, spin, prec) :
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)

    
    numerator = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    denominator = plus_ultra(spin_squared, minus_ultra(r_squared, times_ultra(mpmath.mpf(2.0), r, prec), prec), prec)
    
    fraction = divide_ultra(numerator, denominator, prec)
    result = sqrt_ultra(fraction, prec)
    
    return result
  
def Correction_dphids_ultra(r, M, theta, spin, prec):  
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    
    numerator = times_ultra(mpmath.mpf(2.0), times_ultra(r, times_ultra(spin_squared, sin_squared, prec), prec), prec)
    denominator = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    fraction = divide_ultra(numerator, denominator, prec)
    result_3 = plus_ultra(r_squared, plus_ultra(spin_squared, fraction, prec), prec)
    result_2 = sqrt_ultra(result_3, prec)
    result = times_ultra(sin_theta, result_2, prec)
    
    return result  

def Correction_dths_ultra(r, M, theta, spin, prec):
    
    # Set the precision
    mpmath.mp.dps = prec

    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    r = mpmath.mpf(r)
    theta = mpmath.mpf(theta)

    # Calculate sine and cosine with ultra-precision
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    
    
    result_1 = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    result = sqrt_ultra(result_1, prec)
    
    return result

#################################################################################################################################################################

def calc_deflection(radius_in, theta_in, phi_in, alpha_in, beta_in, prec):
    
    local_start_time = time.time()
    
    # Set the precision
    mpmath.mp.dps = prec

    M = 1.0
    spin = mpmath.mpf("0.999999999999999999999999999999999999999999999999999999999999999999999999999999999")
    # Convert inputs to mpmath.mpf for consistent precision
    M = mpmath.mpf(M)
    spin = mpmath.mpf(spin)
    
    #radius = mpmath.mpf(radius_in)
    theta = mpmath.mpf(theta_in)
    phi = mpmath.mpf(phi_in)
    
    #PI = 3.141592654
    s = mpmath.mpf(0.0)
    ds = 0.01
    #done = 0
    t = mpmath.mpf(0.0)
    r = mpmath.mpf(radius_in)
    alpha = mpmath.mpf(alpha_in)
    beta = mpmath.mpf(beta_in)
    
    sin_theta = sin_ultra(theta, prec)
    cos_theta = cos_ultra(theta, prec)

    # Calculate terms needed for the numerator and denominator
    r_squared = times_ultra(r, r, prec)
    spin_squared = times_ultra(spin, spin, prec)
    sin_squared = times_ultra(sin_theta, sin_theta, prec)
    cos_squared = times_ultra(cos_theta, cos_theta, prec)
    
    #############################################################################################################

    LookDirection = [times_ultra(cos_ultra(alpha, prec), sin_ultra(beta, prec), prec), 
                     times_ultra(sin_ultra(alpha, prec), sin_ultra(beta, prec), prec), 
                     cos_ultra(beta, prec)]
    
    r_direction = [times_ultra(cos_ultra(phi, prec), sin_ultra(theta, prec), prec),
                   times_ultra(sin_ultra(phi, prec), sin_ultra(theta, prec), prec), 
                   cos_ultra(theta, prec)]
    
    phi_direction = [times_ultra(mpmath.mpf(-1.0), sin_ultra(phi, prec), prec),
                     cos_ultra(phi, prec),
                     mpmath.mpf(0.0)]

    theta_direction = [times_ultra(cos_ultra(theta, prec), cos_ultra(phi, prec), prec),
                       times_ultra(cos_ultra(theta, prec), sin_ultra(phi, prec), prec),
                       times_ultra(mpmath.mpf(-1.0), sin_ultra(theta, prec), prec)]
    
    ###############################################################################################################
    
    #drds = np.dot(LookDirection, r_direction) / Correction_drds(r, M, theta, spin)
    
    drds = divide_ultra(dot_ultra(LookDirection, r_direction, prec), Correction_drds_ultra(r, M, theta, spin, prec), prec)
    
    #dphids = np.dot(LookDirection, phi_direction) / Correction_dphids(r, M, theta, spin)
    
    dphids = divide_ultra(dot_ultra(LookDirection, phi_direction, prec), Correction_dphids_ultra(r, M, theta, spin, prec), prec)
    
    #dthds = np.dot(LookDirection, theta_direction) / Correction_dths(r, M, theta, spin)
    
    dthds = divide_ultra(dot_ultra(LookDirection, theta_direction, prec), Correction_dths_ultra(r, M, theta, spin, prec), prec)
    
    #######################################################################################################################################################

    #dtds = (-g_tphi * dphids * 2.0 - np.sqrt(g_tphi * dphids * 2.0 * g_tphi * dphids * 2.0 - 4.0 * g_tt * (g_rr * drds * drds + g_thth * dthds * dthds + g_phiphi * dphids * dphids))) / (2.0 * g_tt)
    
    d2tds2 = mpmath.mpf(0.0)
    d2rds2 = mpmath.mpf(0.0)
    d2thds2 = mpmath.mpf(0.0)
    d2phids2 = mpmath.mpf(0.0)

    d2tds2n = mpmath.mpf(0.0)
    d2rds2n = mpmath.mpf(0.0)
    d2thds2n = mpmath.mpf(0.0)
    d2phids2n = mpmath.mpf(0.0)
    
    # Leapfrog Integration
    times = []
    radii = []
    thetas = []
    phis = []
    
    Status = 0

    ####################################################################################################################################

    for k in range(step_number):
        #if int(k) % 100 == 0:
            #print(f"{int((k/step_number)*100)} %")
        #if k % 100 == 0:
            #print(k)
        ds = 0.01
        
        if times_ultra(r, sin_theta, prec) < mpmath.mpf(1.0):
            ds = times_ultra(ds, times_ultra(r, sin_theta, prec), prec)
            
        alpha = mpmath.mpf(alpha_in)
        beta = mpmath.mpf(beta_in)
        
        sin_theta = sin_ultra(theta, prec)
        cos_theta = cos_ultra(theta, prec)

        # Calculate terms needed for the numerator and denominator
        r_squared = times_ultra(r, r, prec)
        spin_squared = times_ultra(spin, spin, prec)
        sin_squared = times_ultra(sin_theta, sin_theta, prec)
        cos_squared = times_ultra(cos_theta, cos_theta, prec)
            
        Sigma = plus_ultra(r_squared, times_ultra(spin_squared, cos_squared, prec), prec)
    
        Delta = plus_ultra(r_squared, plus_ultra(spin_squared, times_ultra(mpmath.mpf(-2.0), r, prec), prec), prec)

        #g_tt = -(1.0 - (2.0 * M * r) / Sigma)
        g_tt = times_ultra(mpmath.mpf(-1.0), (minus_ultra(mpmath.mpf(1.0), divide_ultra(times_ultra(mpmath.mpf(2.0), r, prec), Sigma, prec), prec)), prec)
        
        #g_tphi = -(((2.0 * M * r) / Sigma) * (spin * np.sin(theta) * np.sin(theta)))
        g_tphi = times_ultra(mpmath.mpf(-1.0), times_ultra(spin, times_ultra(sin_squared, divide_ultra(times_ultra(mpmath.mpf(2.0), r, prec), Sigma, prec), prec), prec), prec)
        
        #g_rr = Sigma / Delta
        g_rr = divide_ultra(Sigma, Delta, prec)
        
        #g_thth = Sigma
        g_thth = Sigma
        
        #g_phiphi = (r * r + spin * spin + (2.0 * M * r * spin * spin * np.sin(theta) * np.sin(theta) / Sigma)) * (np.sin(theta) * np.sin(theta))
        g_phiphi = times_ultra(sin_squared, plus_ultra(r_squared, plus_ultra(spin_squared, divide_ultra(times_ultra(mpmath.mpf(2.0), times_ultra(r, spin_squared, prec), prec), Sigma, prec), prec), prec), prec)
                    
        part1 = times_ultra(mpmath.mpf(-1.0), times_ultra(g_tphi, dphids, prec), prec)
        part11 = times_ultra(power_ultra(g_tphi, 2, prec), power_ultra(dphids, 2, prec), prec)
        aaa = times_ultra(g_rr, power_ultra(drds, 2, prec), prec)
        bbb = times_ultra(g_thth, power_ultra(dthds, 2, prec), prec)
        ccc = times_ultra(g_phiphi, power_ultra(dphids, 2, prec), prec)
        sub_sub = plus_ultra(aaa, plus_ultra(bbb, ccc, prec), prec)
        part22 = times_ultra(mpmath.mpf(-1.0), times_ultra(g_tt, sub_sub, prec), prec)
        sub = plus_ultra(part11, part22, prec)
        
        tolerance = mpmath.mpf('1e-50')
        
        if g_tt > mpmath.mpf(0.0):
            gtt = g_tt
        else:
            gtt = times_ultra(mpmath.mpf(-1.0), g_tt, prec)
        
        if sub < mpmath.mpf(0.0):# or gtt < tolerance:
            print("Fallen into Black Hole!")
            Status = 2
            break            
        
        parenthesis = sqrt_ultra(sub, prec)
        part2 = times_ultra(mpmath.mpf(-1.0), parenthesis, prec)
        numerator = plus_ultra(part1, part2, prec)  
        
        if sub >= mpmath.mpf(0.0):
            dtds = divide_ultra(numerator, g_tt, prec)   

        d2tds2 = fd2tds2(M, r, theta, dtds, drds, dthds, dphids, prec)
        d2rds2 = fd2rds2(M, r, theta, dtds, drds, dthds, dphids, prec)
        d2thds2 = fd2thds2(M, r, theta, dtds, drds, dthds, dphids, prec)
        d2phids2 = fd2phids2(M, r, theta, dtds, drds, dthds, dphids, prec)
        
        t = plus_ultra(t, plus_ultra(times_ultra(dtds, ds, prec), times_ultra(mpmath.mpf(0.5), times_ultra(d2tds2, power_ultra(ds, 2, prec), prec), prec), prec), prec)
        r = plus_ultra(r, plus_ultra(times_ultra(drds, ds, prec), times_ultra(mpmath.mpf(0.5), times_ultra(d2rds2, power_ultra(ds, 2, prec), prec), prec), prec), prec)
        theta = plus_ultra(theta, plus_ultra(times_ultra(dthds, ds, prec), times_ultra(mpmath.mpf(0.5), times_ultra(d2thds2, power_ultra(ds, 2, prec), prec), prec), prec), prec)
        phi = plus_ultra(phi, plus_ultra(times_ultra(dphids, ds, prec), times_ultra(mpmath.mpf(0.5), times_ultra(d2phids2, power_ultra(ds, 2, prec), prec), prec), prec), prec)
        
        times.append(t)
        radii.append(r)
        thetas.append(theta)
        phis.append(phi)

        d2tds2n = fd2tds2(M, r, theta, dtds, drds, dthds, dphids, prec)
        d2rds2n = fd2rds2(M, r, theta, dtds, drds, dthds, dphids, prec)
        d2thds2n = fd2thds2(M, r, theta, dtds, drds, dthds, dphids, prec)
        d2phids2n = fd2phids2(M, r, theta, dtds, drds, dthds, dphids, prec)
        
        dtds = plus_ultra(dtds, times_ultra(mpmath.mpf(0.5), times_ultra(ds, plus_ultra(d2tds2, d2tds2n, prec), prec), prec), prec)
        drds = plus_ultra(drds, times_ultra(mpmath.mpf(0.5), times_ultra(ds, plus_ultra(d2rds2, d2rds2n, prec), prec), prec), prec)
        dthds = plus_ultra(dthds, times_ultra(mpmath.mpf(0.5), times_ultra(ds, plus_ultra(d2thds2, d2thds2n, prec), prec), prec), prec)
        dphids = plus_ultra(dphids, times_ultra(mpmath.mpf(0.5), times_ultra(ds, plus_ultra(d2phids2, d2phids2n, prec), prec), prec), prec)

        # Ray escapes the domain
        if drds > mpmath.mpf(0.0) and r > mpmath.mpf(8.0):
            print(f"Escaped!")
            Status = 1
            break

        # Ray gets to the horizon (or close enough)
        if drds < mpmath.mpf(0.0) and r < plus_ultra(plus_ultra(M, sqrt_ultra(minus_ultra(power_ultra(M, 2, prec), spin_squared, prec), prec), prec), mpmath.mpf(0.001), prec):                      
            print("Fallen in!")
            Status = 2
            break

        s = plus_ultra(s, ds, prec)
        
    local_end_time = time.time()
    print(f"Computation: {round((local_end_time - local_start_time)/60,2)} min")
    print("#############################################################################################")
        
    return times, radii, thetas, phis, Status


def add_zeros_penultimate(s, num_zeros):
    """
    Adds the specified number of '0's to the penultimate position of a string.

    Args:
        s (str): The original string.
        num_zeros (int): The number of '0's to add.

    Returns:
        str: The modified string with the added '0's.
    """
    # Insert '0' * num_zeros at the penultimate position
    return s[:-1] + '0' * num_zeros + s[-1]

def round_mpf(value, decimal_places):
    """
    Rounds an mpmath.mpf object to the specified number of decimal places.

    Args:
        value (mpmath.mpf): The mpf number to round.
        decimal_places (int): The number of decimal places to round to.

    Returns:
        mpmath.mpf: The rounded mpf object.
    """
    # Store the original precision to restore later
    original_dps = 200
    
    # Temporarily increase precision slightly for accurate rounding
    mpmath.mp.dps = decimal_places + 2
    rounded_value = mpmath.nstr(value, decimal_places)

    # Restore original precision
    mpmath.mp.dps = original_dps
    
    return mpmath.mpf(rounded_value)

################################################################################################################################################################################
mpmath.mp.dps = 200
#PI = mpmath.mpf(3.14159265358979323846264338327950288419716939937510)
PI = 3.141592653589793
r_0, theta_0, phi_0, alpha_dir, beta_dir, precision = mpmath.mpf("4.0"), mpmath.mpf("1.57079632679489661923132169163975144209858469968755291048747229615390820314310449931401741267105853399107404325664115332354692230477529111586267970406424055872514205135096926055277986439227"), 0, mpmath.mpf("1.57079632679489661923132169163975144209858469968755291048747229615390820314310449931401741267105853399107404325664115332354692230477529111586267970406424055872514205135096926055277986439227"), 0, 200

#precise_radius = find_precise_r0(r_0, 6)
#print(f"The precise value of r_0 is {precise_radius}")

r_guesses = []
steps = []
decimals_amount = []

def path_finder(r_guess, decimal_precision):
    
    current_decimals = 1
    current_status = 1
    
    counter = 0
    
    step = "0.1"
    
    while current_decimals <= decimal_precision:
        
        while current_status == 1:
            
            counter = counter + 1
            
            r_guess = minus_ultra(r_guess, step, 200)
            r_guess = round_mpf(r_guess, current_decimals + 1)
            
            r_guesses.append(r_guess)
            steps.append(step)
            decimals_amount.append(current_decimals)
            print("Running...")
            print(f"r: {r_guess}") 
            print(f"s: {step}")  
            print(f"decimals: {current_decimals}")
            times, radii, thetas, phis, current_status = calc_deflection(r_guess, theta_0, phi_0, alpha_dir, beta_dir, precision)
            
            """fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x_min, x_max = -5, 5
            y_min, y_max = -5, 5
            z_min, z_max = -5, 5

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # Convert to floats for plotting
            radii_floats = [float(r) for r in radii]
            thetas_floats = [float(theta) for theta in thetas]
            phis_floats = [float(phi) for phi in phis]

            ax.plot(radii_floats * np.sin(thetas_floats) * np.cos(phis_floats), radii_floats * np.sin(thetas_floats) * np.sin(phis_floats), radii_floats * np.cos(thetas_floats), color="red")
            plt.title({current_status})
            
            plt.savefig(f"{counter}.jpg", dpi=100)"""
            
        
        # Check if we need to enter the second loop
        if current_status == 2:
            # Increment decimal precision before entering the second loop
            current_decimals += 1    
            
            
        #current_decimals = current_decimals + 1     
        while current_status == 2:
            counter = counter + 1
            
            old_step = step
            step = add_zeros_penultimate(step, 1)
            steps.append(step) 
            decimals_amount.append(current_decimals)  
                        
            r_guess = minus_ultra(plus_ultra(r_guess, old_step, 200), step, 200)
            r_guess = round_mpf(r_guess, current_decimals + 1)
            r_guesses.append(r_guess)
            print("Running...")
            print(f"r: {r_guess}")
            print(f"s: {step}")
            print(f"decimals: {current_decimals}")
            times, radii, thetas, phis, current_status = calc_deflection(r_guess, theta_0, phi_0, alpha_dir, beta_dir, precision)
            
            """fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x_min, x_max = -5, 5
            y_min, y_max = -5, 5
            z_min, z_max = -5, 5

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # Convert to floats for plotting
            radii_floats = [float(r) for r in radii]
            thetas_floats = [float(theta) for theta in thetas]
            phis_floats = [float(phi) for phi in phis]

            ax.plot(radii_floats * np.sin(thetas_floats) * np.cos(phis_floats), radii_floats * np.sin(thetas_floats) * np.sin(phis_floats), radii_floats * np.cos(thetas_floats), color="red")
            plt.title({current_status})
            plt.savefig(f"{counter}.jpg", dpi=100)"""
            
            
            if current_status == 2:
            
                current_decimals += 1
            
    return r_guess


r_accurate_estimate = path_finder(r_0, 150)

print(f"Accurate r:{r_accurate_estimate}")
print(type(r_accurate_estimate))

times, radii, thetas, phis, Status = calc_deflection(r_accurate_estimate, theta_0, phi_0, alpha_dir, beta_dir, precision)

##################################################################################################################################################

def plot_sphere(ax, radius, center=(0, 0, 0), color="gray"):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_min, x_max = -5, 5
y_min, y_max = -5, 5
z_min, z_max = -5, 5

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

plot_sphere(ax, radius = M + np.sqrt(M * M - spin * spin), center=(0, 0, 0), color="gray")

# Convert to floats for plotting
radii_floats = [float(r) for r in radii]
thetas_floats = [float(theta) for theta in thetas]
phis_floats = [float(phi) for phi in phis]

ax.plot(radii_floats * np.sin(thetas_floats) * np.cos(phis_floats), radii_floats * np.sin(thetas_floats) * np.sin(phis_floats), radii_floats * np.cos(thetas_floats), color="red")

ax.view_init(elev=23, azim=7)

end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the execution time
print(f"Execution time: {round(execution_time* 0.000277778, 2)} h") # to hours * 0.000277778

#plt.show()
#plt.savefig("Path_finder_2.png", dpi=3000)


######################################################################################################

"""# Zip the lists together to create rows
data_rows = zip(r_guesses, steps, decimals_amount)

# Specify the file name
csv_file_name = "Error_tracer.csv"

# Write the data to the CSV file
with open(csv_file_name, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header
    csv_writer.writerow(['r', 'step', 'decimals'])
    
    # Write the data rows
    csv_writer.writerows(data_rows)

print(f"Data has been written to {csv_file_name}")"""