#=============================================================================
# Find JPEG XL library
#=============================================================================
# Find the native JPEG XL headers and libraries.
#
#  JXL_INCLUDE_DIRS - where to find jxl/decode_cxx.h, etc.
#  JXL_LIBRARIES    - List of libraries when using jxl.
#  JXL_FOUND        - True if jxl is found.
#=============================================================================

# Look for the header file.

message(STATUS "Searching for JPEG XL")
find_package(PkgConfig)
if (PkgConfig_FOUND)
    pkg_check_modules(JXL IMPORTED_TARGET libjxl libjxl_cms libjxl_threads)
    if (JXL_FOUND)
        message(STATUS "Found libjxl via pkg-config in `${JXL_LIBRARY_DIRS}`")
    else()
        message(" *****************************************************************************")
        message(" *** No JPEG XL libraries found.                                           ***")
        message(" *** On Ubuntu 23.04 and newer you can install them by executing           ***")
        message(" ***    sudo apt install libjxl-dev                                        ***")
        message(" ***                                                                       ***")
        message(" *** Otherwise, you can find precompiled packages here:                    ***")
        message(" ***    https://github.com/libjxl/libjxl/releases                          ***")
        message(" *****************************************************************************")
    endif()
else()
    message(STATUS "PkgConfig could not be found, JPEG XL support won't be available")
    set(JXL_FOUND 0)
endif()

if(JXL_FOUND)
    set(JXL_TEST_CMAKE_FLAGS
      "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
      "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
      "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

    try_compile(test_for_libjxl_worked
        ${PROJECT_BINARY_DIR}/test_for_libjxl_build
        ${CMAKE_CURRENT_LIST_DIR}/test_for_libjxl
        test_if_libjxl_is_broken
        CMAKE_FLAGS "${JXL_TEST_CMAKE_FLAGS}")

    if(NOT test_for_libjxl_worked)
        set(JXL_FOUND 0)
        message (STATUS "System copy of libjxl is either too old or broken.  Will disable JPEG XL support.")
    endif()
endif()
