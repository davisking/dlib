#ifndef DLIB_sIMD_VSX_Hh_
#define DLIB_sIMD_VSX_Hh_

#include <altivec.h> // VSX
#include <stdlib.h>

/* Returns amount of bytes till previous 16-byte aligned value  */
inline size_t getAlignOffset(const void *p) const {
    return ((size_t) p) & 0x0f;// This is optimised variant of ((size_t) p) % 16
}

#endif //DLIB_sIMD_VSX_Hh_
