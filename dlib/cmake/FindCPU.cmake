include(FindPackageHandleStandardArgs)
include(CheckCSourceRuns)
include(CheckTypeSize)
check_type_size( "void*" SIZE_OF_VOID_PTR)

# Test for AVX2
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
endif()

check_c_source_runs(
"
#include <immintrin.h>
int main()
{
    __m256i a       = _mm256_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4);
    __m256i result  = _mm256_abs_epi32 (a);
    return 0;
}
" HAVE_AVX2)

if (HAVE_AVX2)
    set(AVX2_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for AVX
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-mavx")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
endif()
  
check_c_source_runs(
"
#include <immintrin.h>
int main()
{
    __m256 a = _mm256_set_ps (-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 b = _mm256_set_ps (1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f);
    __m256 result = _mm256_add_ps (a, b);
    return 0;
}" HAVE_AVX)

if (HAVE_AVX)
    set(AVX_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for SSE 4.2
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-msse4.2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
endif()

check_c_source_runs(
"
#include <emmintrin.h>
#include <nmmintrin.h>
int main()
{
    long long a[2] = {  1, 2 };
    long long b[2] = { -1, 3 };
    long long c[2];
    __m128i va = _mm_loadu_si128((__m128i*)a);
    __m128i vb = _mm_loadu_si128((__m128i*)b);
    __m128i vc = _mm_cmpgt_epi64(va, vb);

    _mm_storeu_si128((__m128i*)c, vc);
    if (c[0] == -1LL && c[1] == 0LL)
      return 0;
    else
      return 1;
}" HAVE_SSE42)

if (HAVE_SSE42)
    set(SSE42_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for SSE 4.1
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-msse4.1")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
endif()

check_c_source_runs(
"
#include <emmintrin.h>
#include <smmintrin.h>
int main()
{
    long long a[2] = {  1, 2 };
    long long b[2] = { -1, 2 };
    long long c[2];
    __m128i va = _mm_loadu_si128((__m128i*)a);
    __m128i vb = _mm_loadu_si128((__m128i*)b);
    __m128i vc = _mm_cmpeq_epi64(va, vb);

    _mm_storeu_si128((__m128i*)c, vc);
    if (c[0] == 0LL && c[1] == -1LL)
      return 0;
    else
      return 1;
}" HAVE_SSE41)

if (HAVE_SSE41)
    set(SSE41_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for NEON
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-mfpu=neon")
else()
    set(CMAKE_REQUIRED_FLAGS "")
endif()

check_c_source_runs(
"
#include <stdio.h>
#include <arm_neon.h>

int main(void){
//vector addition 8x8 example.
int r,s;

uint8x8_t a = vdup_n_u8(9);
uint8x8_t b = vdup_n_u8(10);

uint8x8_t dst = a * b;

r = vget_lane_u8( dst, 0);
s = vget_lane_u8( dst, 7);

if (r != 90 || s != 90) {
   return 1;
}

return 0;
}" HAVE_NEON)

if (HAVE_NEON)
    set(NEON_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()


find_package_handle_standard_args(CPU DEFAULT_MSG HAVE_AVX)