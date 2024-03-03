include(FindPackageHandleStandardArgs)
include(CheckCSourceRuns)
include(CheckTypeSize)
check_type_size( "void*" SIZE_OF_VOID_PTR)

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

find_package_handle_standard_args(NEON DEFAULT_MSG NEON_CFLAGS)