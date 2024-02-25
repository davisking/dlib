include(FindPackageHandleStandardArgs)
include(CheckCSourceRuns)
include(CheckTypeSize)
check_type_size( "void*" SIZE_OF_VOID_PTR)

# Test for SSE 2
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-msse2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
endif()

check_c_source_runs(
"
#include <emmintrin.h>
  int main()
  {
    int a[4] = { 1, 2,  3,  4 };
    int b[4] = { 3, 6, -4, -4 };
    int c[4];

    __m128i va = _mm_loadu_si128((__m128i*)a);
    __m128i vb = _mm_loadu_si128((__m128i*)b);
    __m128i vc = _mm_add_epi32(va, vb);

    _mm_storeu_si128((__m128i*)c, vc);
    if (c[0] == 4 && c[1] == 8 && c[2] == -1 && c[3] == 0)
      return 0;
    else
      return 1;
  }
" HAVE_SSE2)

if (HAVE_SSE2)
    list(APPEND SSE_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for SSE 3
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-msse3")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
endif()

check_c_source_runs(
"
#include <emmintrin.h>
  #ifdef _WIN32
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif

  int main()
  {
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float b[4] = { 3.0f, 5.0f, 7.0f, 9.0f };
    float c[4];

    __m128 va = _mm_loadu_ps(a);
    __m128 vb = _mm_loadu_ps(b);
    __m128 vc = _mm_hadd_ps(va, vb);

    _mm_storeu_ps(c, vc);
    if (c[0] == 3.0f && c[1] == 7.0f && c[2] == 8.0f && c[3] == 16.0f)
      return 0;
    else
      return 1;
  }
" HAVE_SSE3)

if (HAVE_SSE3)
    list(APPEND SSE_CFLAGS ${CMAKE_REQUIRED_FLAGS})
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
    list(APPEND SSE_CFLAGS ${CMAKE_REQUIRED_FLAGS})
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
    list(APPEND SSE_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

find_package_handle_standard_args(SSE DEFAULT_MSG SSE_CFLAGS)