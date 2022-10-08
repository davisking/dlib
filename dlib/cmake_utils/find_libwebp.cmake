#=============================================================================
# Find WebP library
# From OpenCV
#=============================================================================
# Find the native WebP headers and libraries.
#
#  WEBP_INCLUDE_DIRS - where to find webp/decode.h, etc.
#  WEBP_LIBRARIES    - List of libraries when using webp.
#  WEBP_FOUND        - True if webp is found.
#=============================================================================

# Look for the header file.

set(WEBP_FOUND False)

find_path(WEBP_INCLUDE_DIR NAMES webp/decode.h)
find_library(WEBP_LIBRARY NAMES webp)

if(WEBP_INCLUDE_DIR AND WEBP_LIBRARY)
    mark_as_advanced(WEBP_INCLUDE_DIR WEBP_LIBRARY)
    set(WEBP_FOUND True)
    set(WEBP_LIBRARIES ${WEBP_LIBRARY})
    set(WEBP_INCLUDE_DIRS ${WEBP_INCLUDE_DIR})
endif()
