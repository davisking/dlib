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

unset(JXL_FOUND)

find_path(JXL_INCLUDE_DIR NAMES jxl/decode_cxx.h jxl/encode_cxx.h)

if(NOT JXL_INCLUDE_DIR)
    unset(JXL_FOUND)
else()
    mark_as_advanced(JXL_INCLUDE_DIR)

    # Look for the library
    find_library(JXL_LIBRARY NAMES jxl)
    # handle the QUIETLY and REQUIRED arguments and set JXL_FOUND to TRUE if
    # all listed variables are TRUE
    include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
    find_package_handle_standard_args(JXL DEFAULT_MSG JXL_LIBRARY JXL_INCLUDE_DIR)

    set(JXL_LIBRARIES ${JXL_LIBRARY})
    set(JXL_INCLUDE_DIRS ${JXL_INCLUDE_DIR})
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
