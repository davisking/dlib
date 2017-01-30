#ifndef DLIB_sIMD_VSX_Hh_
#define DLIB_sIMD_VSX_Hh_

#include <altivec.h> // VSX
#include <stdlib.h>

/* Returns amount of bytes till previous 16-byte aligned value  */
inline size_t vsx_getAlignOffset(const void *p) {
    return ((size_t) p) % 16;
}

#endif //DLIB_sIMD_VSX_Hh_
