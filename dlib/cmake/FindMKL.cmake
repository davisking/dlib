# We're going to be opiniated and not allow the use of:
# - libmkl_rt.so            : 
#       as this requires users to manually set both the interface layer and threading layer at runtime

# - libmkl_intel_thread.so  : 
#       as this dynamically loads libiomp5.so at runtime (using dlopen) 
#       and will very likely conflict with any software upstream that uses OpenMP (which could be a lot)
#       for example, ffmpeg libs.

# - libmkl_gnu_thread.so    :
#       as this dynamically loads libgomp.so at runtime (using dlopen). 
#       This could be ok, but better to use a threading layer that doesn't depend on OMP at all.

include(FindPackageHandleStandardArgs)
include(CheckTypeSize)
check_type_size( "void*" SIZE_OF_VOID_PTR)

set(mkl_search_path
    /opt/intel/oneapi/mkl/latest)

if (SIZE_OF_VOID_PTR EQUAL 8)
    set(mkl_lib_suffix lib)
    set(mkl_intel_name mkl_intel_lp64)
else()
    set(mkl_lib_suffix lib32)
    set(mkl_intel_name mkl_intel)
endif()

find_path(MKL_INCLUDE_DIR       NAMES mkl_version.h     PATHS ${mkl_search_path} PATH_SUFFIXES mkl include)
find_library(MKL_CORE_LIB       NAMES mkl_core          PATHS ${mkl_search_path} PATH_SUFFIXES ${mkl_lib_suffix})
find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential    PATHS ${mkl_search_path} PATH_SUFFIXES ${mkl_lib_suffix})
find_library(MKL_TBB_LIB        NAMES mkl_tbb_thread    PATHS ${mkl_search_path} PATH_SUFFIXES ${mkl_lib_suffix})
find_library(MKL_INTEL_LIB      NAMES ${mkl_intel_name} PATHS ${mkl_search_path} PATH_SUFFIXES ${mkl_lib_suffix})  

mark_as_advanced(MKL_INCLUDE_DIR)
mark_as_advanced(MKL_CORE_LIB)
mark_as_advanced(MKL_INTEL_LIB)
mark_as_advanced(MKL_SEQUENTIAL_LIB MKL_TBB_LIB)

add_library(mkl_core UNKNOWN IMPORTED)
set_target_properties(mkl_core PROPERTIES 
    IMPORTED_LOCATION               ${MKL_CORE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES   ${MKL_INCLUDE_DIR})

add_library(mkl_intel UNKNOWN IMPORTED)
set_target_properties(mkl_intel PROPERTIES 
    IMPORTED_LOCATION               ${MKL_INTEL_LIB}
    INTERFACE_INCLUDE_DIRECTORIES   ${MKL_INCLUDE_DIR})

add_library(mkl_sequential UNKNOWN IMPORTED)
set_target_properties(mkl_sequential PROPERTIES 
    IMPORTED_LOCATION               ${MKL_SEQUENTIAL_LIB}
    INTERFACE_INCLUDE_DIRECTORIES   ${MKL_INCLUDE_DIR})

add_library(mkl_tbb UNKNOWN IMPORTED)
set_target_properties(mkl_tbb PROPERTIES 
    IMPORTED_LOCATION               ${MKL_TBB_LIB}
    INTERFACE_INCLUDE_DIRECTORIES   ${MKL_INCLUDE_DIR})

add_library(mkl::sequential INTERFACE IMPORTED)
target_link_libraries(mkl::sequential INTERFACE mkl_intel mkl_sequential mkl_core)

add_library(mkl::tbb INTERFACE IMPORTED)
target_link_libraries(mkl::tbb INTERFACE mkl_intel mkl_tbb mkl_core)

find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR MKL_CORE_LIB MKL_INTEL_LIB MKL_SEQUENTIAL_LIB MKL_TBB_LIB)