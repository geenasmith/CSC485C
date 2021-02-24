/**
 * Toy example to illustrate SIMD/vectorisation.
 *
 * Calculates the average size of set of 3d vectors.
 * Example input: {{1,1,1},{2,2,2}}
 * I.e., 1 vector of size (3*1^2)^-0.5 and 1 vector of size (3*2^2)^0.5
 * Example output: ( sqrt(3) + sqrt(12) ) / 2 = 2.59808
 */


#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX
#include "math.h" 	   // sqrtf()
#include <chrono>      // timing library
#include <cassert>     // assert()
#include <iostream>

#include <vector>

namespace AoS{

// Note that for SIMD, it is *imperative* that data is aligned,
// indicated here by `__attribute__ ((aligned(16)))`; otherwise,
// moving data into SIMD registers will be expensive (if copied
// one-by-one as in __mm_setr_ps()) or a segmentation fault (if
// indicated by a pointer, as in __mm_load_ps()).
// For SSE4.2 (i.e., 128-bit SIMD), 16-byte alignment is required;
// for AVX (i.e., 256-bit SIMD), 32-byte alignment is required.
// In this struct, we assume 128-bit SIMD, because we can only
// make use of up to 3 SIMD lanes, anyway.
struct __attribute__ ((aligned (16))) direction
{
	float x;
	float y;
	float z;
};

#include "3d-vectors.hpp" // statically-generated, 16B-aligned `const direction dirs[]`

namespace nosimd {

/** Compute the average magnitude/length of the global set of 3d direction vectors. */
float average_vector_length()
{
    auto total = 0.0f;
    auto const n = sizeof( dirs ) / sizeof( direction );

    // standard loop; could be parallelised with multicore and/or unrolled for ILP
    // as in previous lectures
    for( auto i = 0lu; i < n; ++i )
    {
        total += sqrtf( dirs[ i ].x * dirs[ i ].x +
                        dirs[ i ].y * dirs[ i ].y +
                        dirs[ i ].z * dirs[ i ].z );
    }

    return total / n;
}
} // namespace nosimd
namespace simd {


/**
 * You should complete this function using SIMD intrinsics so that it produces the same output
 * as nosimd::average_vector_length().
 */
float average_vector_length()
{
    // To hand-code SIMD, we basically write assembly code.
    // Specifically, we write *intrinsics*.
    // For a complete list, see: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
    auto total = 0.0f;
    auto const n = sizeof( dirs ) / sizeof( direction );

        // // standard loop; could be parallelised with multicore and/or unrolled for ILP
    // // as in previous lectures
    for( auto i = 0lu; i < n; ++i )
    {
        // __m128 is a data type, but refers to a 128-bit register.
        // Contrast this with a float, which is 32-bits. I.e., an __m128
        // is equivalent to a float[4]. (You could treat them the same with
        // a reinterpret_cast<>().)
        __m128 vector4{};

        // Loads 4 floats starting at the given address into a __m128 variable/register
        // Note that the struct is laid out as <x,y,z>, so we start loading from the x-coord.
        // This loads 4!! values, so it will also load the x-value of the next point, too.
        // It is unavoidable, because we need 128 bits to fill the register/data type.
        // It returns an __m128.

        vector4 = _mm_load_ps( &( dirs[ i ].x ) );


        // This is equivalent to vector[4][0]*vector[4][0], ..., vector4[3]*vector[4][3],
        // i.e., it multiplies each element of the vector piecewise.
        // The result goes into the packed __m128 data type/register, and it is all done in
        // *one* (SSE4) instruction. This provides both parallelism and a lower instruction count
        // You may notice your IPC go down when using SIMD, because your instructions decrease.

        vector4 = _mm_mul_ps( vector4, vector4 );
        
        // // This is equivalent to vector[4][0]+vector[4][0], ..., vector4[3]+vector[4][3],
        // // i.e., it adds each element of the vector piecewise.
        // // The result goes into the packed __m128 data type/register, and it is all done in
        // // *one* (SSE4) instruction.
        // _mm_add_ps( vector4, vector4 );

        // Here we cast the __m128 data type in a float[4] so that we can access
        // the values individually. (This technique is generally called "type punning"
        // and shouldn't be done in modern code without a reinterpret_cast<>() to flag it.)
        auto vec = reinterpret_cast< float const * >( &vector4 );

        total += sqrt(vec[0] + vec[1] + vec[2]);
        // std::cout << vec[ 0 ] << "\t" << vec[ 1 ] << "\t" << vec[ 2 ] << "\t" << vec[ 3 ] << std::endl;


    }


    return total / n;
}

} // namespace simd
} // namespace AoS



int main()
{
    auto sum = 0.0;
    auto const num_trials = 20000u;

    auto const start_time = std::chrono::system_clock::now();

    for( auto i = 0u; i < num_trials; ++i )
    {
        // Toggle namespaces to get nosimd:: or simd:: version
        // sum += AoS::nosimd::average_vector_length();
        sum += AoS::simd::average_vector_length();
    }

    auto const end_time = std::chrono::system_clock::now();
    auto const elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

    std::cout << "answer: " << ( sum  / static_cast< float >( num_trials ) ) << std::endl;
    std::cout << "time: " << ( elapsed_time.count() / static_cast< float >( num_trials ) ) << " us" << std::endl;
    return 0;
}
