#include <dlib/simd/simd_check.h>

#if defined(DLIB_CONFIG_TEST_EXPECT_AVX) && !defined(DLIB_HAVE_AVX)
#error "The installed dlib package did not enable AVX for its consumer"
#endif

int main()
{
    return 0;
}
