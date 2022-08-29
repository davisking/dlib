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

unset(WEBP_FOUND)

find_path(WEBP_INCLUDE_DIR NAMES webp/decode.h)

if(NOT WEBP_INCLUDE_DIR)
    unset(WEBP_FOUND)
else()
    mark_as_advanced(WEBP_INCLUDE_DIR)

    # Look for the library.
    find_library(WEBP_LIBRARY NAMES webp)
    mark_as_advanced(WEBP_LIBRARY)

    # handle the QUIETLY and REQUIRED arguments and set WEBP_FOUND to TRUE if
    # all listed variables are TRUE
    include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    find_package_handle_standard_args(WebP DEFAULT_MSG WEBP_LIBRARY WEBP_INCLUDE_DIR)

    set(WEBP_LIBRARIES ${WEBP_LIBRARY})
    set(WEBP_INCLUDE_DIRS ${WEBP_INCLUDE_DIR})
endif()

if(WEBP_FOUND)
    set(WEBP_TEST_CMAKE_FLAGS
      "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
      "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
      "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

    try_compile(test_for_libwebp_worked
        ${PROJECT_BINARY_DIR}/test_for_libwebp_build
        ${CMAKE_CURRENT_LIST_DIR}/test_for_libwebp
        test_if_libwebp_is_broken
        CMAKE_FLAGS "${WEBP_TEST_CMAKE_FLAGS}")

    if(NOT test_for_libwebp_worked)
        set(WEBP_FOUND 0)
        message (STATUS "System copy of libwebp is either too old or broken.  Will disable WebP support.")
    endif()
endif()
