
#include <immintrin.h>

int main()
{
    unsigned short rdrand16;
    _rdrand16_step(&rdrand16);
    unsigned int rdrand32;
    _rdrand32_step(&rdrand32);
#ifdef __x86_64__
    unsigned long long rdrand64;
    _rdrand64_step(&rdrand64);
#endif
    return 0;
}

// ------------------------------------------------------------------------------------

