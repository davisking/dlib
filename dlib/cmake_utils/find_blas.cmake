#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
#
# This cmake file tries to find installed BLAS and LAPACK libraries.  
# It looks for an installed copy of the Intel MKL library first and then
# attempts to find some other BLAS and LAPACK libraries if you don't have 
# the Intel MKL.
#
#  mkl_seq_found              - True if the Intel MKL sequential library is available
#  mkl_tbb_found              - True if the Intel MKL tbb library is available
#  mkl_thread_found           - True if the Intel MKL thread library is available
#  mkl_found                  - True if at least one of (mkl_seq_found,mkl_tbb_found,mkl_thread_found) is true
#  mkl_include_dir            - MKL include directory
#  mkl_libraries_sequential   - MKL sequential libraries if found
#  mkl_libraries_tbb          - MKL tbb libraries if found
#  mkl_libraries_thread       - MKL thread libraries if found
#  blas_found                 - True if BLAS is available
#  blas_libraries             - link against these to use BLAS library 
#  lapack_found               - True if LAPACK is available
#  lapack_libraries           - link against these to use LAPACK library 

include(CheckTypeSize)
include(CheckFunctionExists)

# setting this makes CMake allow normal looking if else statements (TODO: check if this is still necessary)
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

set(mkl_found                 0)
SET(mkl_seq_found             0)
SET(mkl_tbb_found             0)
SET(mkl_thread_found          0)
SET(blas_found                0)
SET(lapack_found              0)
SET(lapack_with_underscore    0)
SET(lapack_without_underscore 0)

SET(mkl_search_path_unix_64 
   /opt/intel/oneapi/mkl/latest/lib
   /opt/intel/oneapi/tbb/latest/lib
   /opt/intel/oneapi/compiler/latest/lib
   /opt/intel/mkl/*/lib/em64t
   /opt/intel/mkl/lib/intel64
   /opt/intel/lib/intel64
   /opt/intel/mkl/lib
   /opt/intel/tbb/*/lib/em64t/gcc4.7
   /opt/intel/tbb/lib/intel64/gcc4.7
   /opt/intel/tbb/lib/gcc4.7)

SET(mkl_search_path_unix_32
   /opt/intel/oneapi/mkl/latest/lib/ia32
   /opt/intel/mkl/*/lib/32
   /opt/intel/mkl/lib/ia32
   /opt/intel/lib/ia32
   /opt/intel/tbb/*/lib/32/gcc4.7
   /opt/intel/tbb/lib/ia32/gcc4.7)

set(mkl_include_search_path_unix
   /opt/intel/oneapi/mkl/latest/include
   /opt/intel/mkl/include
   /opt/intel/include)

set(mkl_search_path_win_64
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/lib/intel64" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/tbb/lib/intel64/vc14" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/compiler/lib/intel64" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/vc14"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/vc_mt"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64" 
   "C:/Program Files (x86)/Intel/Composer XE/mkl/lib/intel64"
   "C:/Program Files (x86)/Intel/Composer XE/tbb/lib/intel64/vc14"
   "C:/Program Files (x86)/Intel/Composer XE/compiler/lib/intel64"
   "C:/Program Files/Intel/Composer XE/mkl/lib/intel64"
   "C:/Program Files/Intel/Composer XE/tbb/lib/intel64/vc14"
   "C:/Program Files/Intel/Composer XE/compiler/lib/intel64"
   "C:/Program Files (x86)/Intel/oneAPI/mkl/*/lib"
   "C:/Program Files (x86)/Intel/oneAPI/compiler/*/lib"
   "C:/Program Files (x86)/Intel/oneAPI/tbb/*/lib"
   "C:/Program Files (x86)/Intel/oneAPI/mkl/*/lib/intel64"
   "C:/Program Files (x86)/Intel/oneAPI/compiler/*/windows/compiler/lib/intel64_win")

set(mkl_search_path_win_32
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/lib/ia32" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/tbb/lib/ia32/vc14" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/compiler/lib/ia32"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/ia32" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/ia32/vc14" 
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/ia32/vc_mt"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/lib/ia32"
   "C:/Program Files (x86)/Intel/Composer XE/mkl/lib/ia32"
   "C:/Program Files (x86)/Intel/Composer XE/tbb/lib/ia32/vc14"
   "C:/Program Files (x86)/Intel/Composer XE/compiler/lib/ia32"
   "C:/Program Files/Intel/Composer XE/mkl/lib/ia32"
   "C:/Program Files/Intel/Composer XE/tbb/lib/ia32/vc14"
   "C:/Program Files/Intel/Composer XE/compiler/lib/ia32"
   "C:/Program Files (x86)/Intel/oneAPI/mkl/*/lib/ia32"
   "C:/Program Files (x86)/Intel/oneAPI/compiler/*/windows/compiler/lib/ia32_win")

set(mkl_redist_path_win_64
   "C:/Program Files (x86)/Intel/oneAPI/compiler/*/bin"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler" 
   "C:/Program Files (x86)/Intel/oneAPI/compiler/*/windows/redist/intel64_win/compiler")

set(mkl_redist_path_win_32
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/compiler"
   "C:/Program Files (x86)/Intel/oneAPI/compiler/*/windows/redist/ia32_win/compiler")

set(mkl_include_search_path_win
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/include"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/compiler/include"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include"
   "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/include"
   "C:/Program Files (x86)/Intel/Composer XE/mkl/include"
   "C:/Program Files (x86)/Intel/Composer XE/compiler/include"
   "C:/Program Files/Intel/Composer XE/mkl/include"
   "C:/Program Files/Intel/Composer XE/compiler/include"
   "C:/Program Files (x86)/Intel/oneAPI/mkl/*/include")

set(CMAKE_REQUIRED_QUIET_SAV ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET     true)
check_type_size("void*" SIZE_OF_VOID_PTR)
set(CMAKE_REQUIRED_QUIET     ${CMAKE_REQUIRED_QUIET_SAV})

if (UNIX OR MINGW)
   set(mkl_include_search_path ${mkl_include_search_path_unix})
   if (SIZE_OF_VOID_PTR EQUAL 8)
      set(mkl_search_path ${mkl_search_path_unix_64})
      set(mkl_intel_name mkl_intel_lp64)
   else()
      set(mkl_search_path ${mkl_search_path_unix_32})
      set(mkl_intel_name mkl_intel)
   endif()
elseif(WIN32)
   set(mkl_include_search_path ${mkl_include_search_path_win})
   if (SIZE_OF_VOID_PTR EQUAL 8)
      set(mkl_search_path ${mkl_search_path_win_64})
      set(mkl_intel_name mkl_intel_lp64)
   else()
      set(mkl_search_path ${mkl_search_path_win_32})
      set(mkl_intel_name mkl_intel_c)
   endif()
endif()

# Search for MKL
find_path(    mkl_include_dir    NAMES mkl_version.h     HINTS ${mkl_include_search_path})
find_library( mkl_core           NAMES mkl_core          HINTS ${mkl_search_path})
find_library( mkl_tbb_thread     NAMES mkl_tbb_thread    HINTS ${mkl_search_path})
find_library( mkl_sequential     NAMES mkl_sequential    HINTS ${mkl_search_path})
find_library( mkl_thread         NAMES mkl_intel_thread  HINTS ${mkl_search_path})
find_library( mkl_intel          NAMES ${mkl_intel_name} HINTS ${mkl_search_path})
find_library( mkl_tbb            NAMES tbb               HINTS ${mkl_search_path})
find_library( mkl_iomp           NAMES iomp5             HINTS ${mkl_search_path}) 

mark_as_advanced(mkl_include_dir)
mark_as_advanced(mkl_core)
mark_as_advanced(mkl_tbb_thread)
mark_as_advanced(mkl_sequential)
mark_as_advanced(mkl_thread)
mark_as_advanced(mkl_intel)
mark_as_advanced(mkl_tbb)
mark_as_advanced(mkl_iomp)

if (mkl_include_dir AND mkl_intel AND mkl_sequential AND mkl_core)
   message(STATUS "Found MKL sequential")
   SET(mkl_seq_found 1)
   SET(mkl_libraries_sequential ${mkl_intel} ${mkl_sequential} ${mkl_core})
endif()

if (mkl_include_dir AND mkl_intel AND mkl_tbb_thread AND mkl_core AND mkl_tbb)
   message(STATUS "Found MKL tbb")
   SET(mkl_tbb_found 1)
   SET(mkl_libraries_tbb ${mkl_intel} ${mkl_tbb_thread} ${mkl_core} ${mkl_tbb})
endif()

if (mkl_include_dir AND mkl_intel AND mkl_thread AND mkl_core AND mkl_iomp)
   message(STATUS "Found MKL thread")
   SET(mkl_thread_found 1)
   set(mkl_libraries_thread ${mkl_intel} ${mkl_thread} ${mkl_core} ${mkl_iomp})
endif()

if (mkl_seq_found OR mkl_tbb_found OR mkl_thread_found)
   set(mkl_found 1)
   return()
endif()

# Search for BLAS - FindBLAS
set(BLA_PREFER_PKGCONFIG 1)
find_package(BLAS)

if (BLAS_FOUND)
   message(STATUS "Found BLAS library")
   set(blas_found       1)
   set(blas_libraries   ${BLAS_LIBRARIES})
endif()

# Search for LAPACK - FindLAPACK
set(BLA_PREFER_PKGCONFIG 1)
find_package(LAPACK)

if (LAPACK_FOUND)
   message(STATUS "Found LAPACK library")
   set(lapack_found     1)
   set(lapack_libraries ${LAPACK_LIBRARIES})
endif()