# It is assumed that find_package(MKL) as been called already

# Order of preference:
#   - pkg-config
#   - openblas
#   - atlas
#   - blas & lapack

include(FindPackageHandleStandardArgs)
include(CheckFunctionExists)
include(CheckLibraryExists)
include(CheckFortranFunctionExists)
find_package(PkgConfig)

# pkg-config
if (PKG_CONFIG_FOUND)
    pkg_check_modules(BLAS IMPORTED_TARGET blas)
    pkg_check_modules(LAPACK IMPORTED_TARGET lapack)

    if (BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "Found BLAS using pkg-config: ${BLAS_LINK_LIBRARIES}")
        add_library(BLAS::BLAS ALIAS PkgConfig::BLAS)
    endif()

    if (LAPACK_FOUND)
        message(STATUS "Found LAPACK using pkg-config: ${LAPACK_LINK_LIBRARIES}")
        add_library(LAPACK::LAPACK ALIAS PkgConfig::LAPACK)
    endif()
endif()

# openblas
if (NOT BLAS_FOUND)
    find_library(OPENBLAS_LIB NAMES openblas PATH_SUFFIXES openblas-pthread openblas-serial openblas-base)

    if (OPENBLAS_LIB)
        message(STATUS "Found openblas")
        set(BLAS_FOUND true)
        set(BLAS_LINK_LIBRARIES ${OPENBLAS_LIB})
        add_library(BLAS::BLAS UNKNOWN IMPORTED)
        set_target_properties(BLAS::BLAS PROPERTIES 
            IMPORTED_LOCATION ${OPENBLAS_LIB})
        
        set(CMAKE_REQUIRED_LIBRARIES ${OPENBLAS_LIB})
        check_function_exists(sgetrf_single OPENBLAS_HAS_LAPACK)
        if (OPENBLAS_HAS_LAPACK)
            message(STATUS "Found openblas's built in LAPACK")
            set(LAPACK_FOUND true)
            set(LAPACK_LINK_LIBRARIES ${OPENBLAS_LIB})
            add_library(LAPACK::LAPACK UNKNOWN IMPORTED)
            set_target_properties(LAPACK::LAPACK PROPERTIES 
                IMPORTED_LOCATION ${OPENBLAS_LIB})
        endif()
    endif()
endif()

find_package_handle_standard_args(CBLAS DEFAULT_MSG BLAS_LINK_LIBRARIES LAPACK_LINK_LIBRARIES)