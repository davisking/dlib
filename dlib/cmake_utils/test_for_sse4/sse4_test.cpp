
#include <xmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h>
#include <smmintrin.h> // SSE4

int main()
{
    __m128 x;
    x = _mm_set1_ps(1.23);
    x = _mm_ceil_ps(x);
    return 0;
}

// ------------------------------------------------------------------------------------

