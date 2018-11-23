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
#  blas_found               - True if BLAS is available
#  lapack_found             - True if LAPACK is available
#  found_intel_mkl          - True if the Intel MKL library is available
#  found_intel_mkl_headers  - True if Intel MKL headers are available
#  blas_libraries           - link against these to use BLAS library 
#  lapack_libraries         - link against these to use LAPACK library 
#  mkl_libraries            - link against these to use the MKL library
#  mkl_include_dir          - add to the include path to use the MKL library
#  openmp_libraries         - Set to Intel's OpenMP library if and only if we
#                             find the MKL.

# setting this makes CMake allow normal looking if else statements
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

SET(blas_found 0)
SET(lapack_found 0)
SET(found_intel_mkl 0)
SET(found_intel_mkl_headers 0)
SET(lapack_with_underscore 0)
SET(lapack_without_underscore 0)

message(STATUS "Searching for BLAS and LAPACK")

if (UNIX OR MINGW)
   message(STATUS "Searching for BLAS and LAPACK")

   if (BUILDING_MATLAB_MEX_FILE)
      # # This commented out stuff would link directly to MATLAB's built in
      # BLAS and LAPACK. But it's better to not link to anything and do a
      #find_library(MATLAB_BLAS_LIBRARY mwblas  PATHS ${MATLAB_LIB_FOLDERS} )
      #find_library(MATLAB_LAPACK_LIBRARY mwlapack  PATHS ${MATLAB_LIB_FOLDERS} )
      #if (MATLAB_BLAS_LIBRARY AND MATLAB_LAPACK_LIBRARY)
      #    add_subdirectory(external/cblas)
      #    set(blas_libraries  ${MATLAB_BLAS_LIBRARY} cblas  )
      #    set(lapack_libraries  ${MATLAB_LAPACK_LIBRARY} )
      #    set(blas_found 1)
      #    set(lapack_found 1)
      #    message(STATUS "Found MATLAB's BLAS and LAPACK libraries")
      #endif()

      # We need cblas since MATLAB doesn't provide cblas symbols.
      add_subdirectory(external/cblas)
      set(blas_libraries  cblas  )
      set(blas_found 1)
      set(lapack_found 1)
      message(STATUS "Will link with MATLAB's BLAS and LAPACK at runtime (hopefully!)")


      ## Don't try to link to anything other than MATLAB's own internal blas
      ## and lapack libraries because doing so generally upsets MATLAB.  So
      ## we just end here no matter what.
      return()
   endif()

   # First, search for libraries via pkg-config, which is the cleanest path
   find_package(PkgConfig)
   pkg_check_modules(BLAS_REFERENCE cblas)
   pkg_check_modules(LAPACK_REFERENCE lapack)
   if (BLAS_REFERENCE_FOUND AND LAPACK_REFERENCE_FOUND)
      set(blas_libraries "${BLAS_REFERENCE_LDFLAGS}")
      set(lapack_libraries "${LAPACK_REFERENCE_LDFLAGS}")
      set(blas_found 1)
      set(lapack_found 1)
      set(REQUIRES_LIBS "${REQUIRES_LIBS} cblas lapack")
      message(STATUS "Found BLAS and LAPACK via pkg-config")
      return()
   endif()

   include(CheckTypeSize)
   check_type_size( "void*" SIZE_OF_VOID_PTR)

   if (SIZE_OF_VOID_PTR EQUAL 8)
      set( mkl_search_path
         /opt/intel/mkl/*/lib/em64t
         /opt/intel/mkl/lib/intel64
         /opt/intel/lib/intel64
         /opt/intel/mkl/lib
         /opt/intel/tbb/*/lib/em64t/gcc4.7
         /opt/intel/tbb/lib/intel64/gcc4.7
         /opt/intel/tbb/lib/gcc4.7
         )

      find_library(mkl_intel mkl_intel_lp64 ${mkl_search_path})
      mark_as_advanced(mkl_intel)
   else()
      set( mkl_search_path
         /opt/intel/mkl/*/lib/32
         /opt/intel/mkl/lib/ia32
         /opt/intel/lib/ia32
         /opt/intel/tbb/*/lib/32/gcc4.7
         /opt/intel/tbb/lib/ia32/gcc4.7
         )

      find_library(mkl_intel mkl_intel ${mkl_search_path})
      mark_as_advanced(mkl_intel)
   endif()

   include(CheckLibraryExists)

   # Get mkl_include_dir
   set(mkl_include_search_path
      /opt/intel/mkl/include
      /opt/intel/include
      )
   find_path(mkl_include_dir mkl_version.h ${mkl_include_search_path})
   mark_as_advanced(mkl_include_dir)

   if(NOT DLIB_USE_MKL_SEQUENTIAL AND NOT DLIB_USE_MKL_WITH_TBB)
      # Search for the needed libraries from the MKL.  We will try to link against the mkl_rt
      # file first since this way avoids linking bugs in some cases.
      find_library(mkl_rt mkl_rt ${mkl_search_path})
      find_library(openmp_libraries iomp5 ${mkl_search_path}) 
      mark_as_advanced(mkl_rt  openmp_libraries)
      # if we found the MKL 
      if (mkl_rt)
         set(mkl_libraries  ${mkl_rt} )
         set(blas_libraries  ${mkl_rt} )
         set(lapack_libraries  ${mkl_rt} )
         set(blas_found 1)
         set(lapack_found 1)
         set(found_intel_mkl 1)
         message(STATUS "Found Intel MKL BLAS/LAPACK library")
      endif()
   endif()
   

   if (NOT found_intel_mkl)
      # Search for the needed libraries from the MKL.  This time try looking for a different
      # set of MKL files and try to link against those.
      find_library(mkl_core mkl_core ${mkl_search_path})
      set(mkl_libs ${mkl_intel} ${mkl_core})
      mark_as_advanced(mkl_libs mkl_intel mkl_core)

      if (DLIB_USE_MKL_WITH_TBB)
         find_library(mkl_tbb_thread mkl_tbb_thread ${mkl_search_path})
         find_library(mkl_tbb tbb ${mkl_search_path})
         mark_as_advanced(mkl_tbb_thread mkl_tbb)
         list(APPEND mkl_libs ${mkl_tbb_thread} ${mkl_tbb})
      elseif (DLIB_USE_MKL_SEQUENTIAL)
         find_library(mkl_sequential mkl_sequential ${mkl_search_path})
         mark_as_advanced(mkl_sequential)
         list(APPEND mkl_libs ${mkl_sequential})
      else()
         find_library(mkl_thread mkl_intel_thread ${mkl_search_path})
         find_library(mkl_iomp iomp5 ${mkl_search_path})
         find_library(mkl_pthread pthread ${mkl_search_path})
         mark_as_advanced(mkl_thread mkl_iomp mkl_pthread)
         list(APPEND mkl_libs ${mkl_thread} ${mkl_iomp} ${mkl_pthread})
      endif()
   
      # If we found the MKL 
      if (mkl_intel AND mkl_core AND ((mkl_tbb_thread AND mkl_tbb) OR (mkl_thread AND mkl_iomp AND mkl_pthread) OR mkl_sequential))
         set(mkl_libraries ${mkl_libs})
         set(blas_libraries ${mkl_libs})
         set(lapack_libraries ${mkl_libs})
         set(blas_found 1)
         set(lapack_found 1)
         set(found_intel_mkl 1)
         message(STATUS "Found Intel MKL BLAS/LAPACK library")
      endif()
   endif()

   if (found_intel_mkl AND mkl_include_dir)
      set(found_intel_mkl_headers 1)
   endif()

   # try to find some other LAPACK libraries if we didn't find the MKL
   set(extra_paths
      /usr/lib64
      /usr/lib64/atlas-sse3
      /usr/lib64/atlas-sse2
      /usr/lib64/atlas
      /usr/lib
      /usr/lib/atlas-sse3
      /usr/lib/atlas-sse2
      /usr/lib/atlas
      /usr/lib/openblas-base
      /opt/OpenBLAS/lib
      $ENV{OPENBLAS_HOME}/lib
      )

   INCLUDE (CheckFunctionExists)

   if (NOT blas_found)
      find_library(cblas_lib openblas PATHS ${extra_paths})
      if (cblas_lib)
         set(blas_libraries ${cblas_lib})
         set(blas_found 1)
         message(STATUS "Found OpenBLAS library")
         set(CMAKE_REQUIRED_LIBRARIES ${blas_libraries})
         # If you compiled OpenBLAS with LAPACK in it then it should have the
         # sgetrf_single function in it.  So if we find that function in
         # OpenBLAS then just use OpenBLAS's LAPACK. 
         CHECK_FUNCTION_EXISTS(sgetrf_single OPENBLAS_HAS_LAPACK)
         if (OPENBLAS_HAS_LAPACK)
            message(STATUS "Using OpenBLAS's built in LAPACK")
            # set(lapack_libraries gfortran) 
            set(lapack_found 1)
         endif()
      endif()
      mark_as_advanced( cblas_lib)
   endif()


   if (NOT lapack_found)
      find_library(lapack_lib NAMES lapack lapack-3 PATHS ${extra_paths})
      if (lapack_lib)
         set(lapack_libraries ${lapack_lib})
         set(lapack_found 1)
         message(STATUS "Found LAPACK library")
      endif()
      mark_as_advanced( lapack_lib)
   endif()


   # try to find some other BLAS libraries if we didn't find the MKL

   if (NOT blas_found)
      find_library(atlas_lib atlas PATHS ${extra_paths})
      find_library(cblas_lib cblas PATHS ${extra_paths})
      if (atlas_lib AND cblas_lib)
         set(blas_libraries ${atlas_lib} ${cblas_lib})
         set(blas_found 1)
         message(STATUS "Found ATLAS BLAS library")
      endif()
      mark_as_advanced( atlas_lib cblas_lib)
   endif()

   # CentOS 7 atlas
   if (NOT blas_found)
      find_library(tatlas_lib tatlas PATHS ${extra_paths})
      find_library(satlas_lib satlas PATHS ${extra_paths})
      if (tatlas_lib AND satlas_lib )
         set(blas_libraries ${tatlas_lib} ${satlas_lib})
         set(blas_found 1)
         message(STATUS "Found ATLAS BLAS library")
      endif()
      mark_as_advanced( tatlas_lib satlas_lib)
   endif()


   if (NOT blas_found)
      find_library(cblas_lib cblas PATHS ${extra_paths})
      if (cblas_lib)
         set(blas_libraries ${cblas_lib})
         set(blas_found 1)
         message(STATUS "Found CBLAS library")
      endif()
      mark_as_advanced( cblas_lib)
   endif()


   if (NOT blas_found)
      find_library(generic_blas blas PATHS ${extra_paths})
      if (generic_blas)
         set(blas_libraries ${generic_blas})
         set(blas_found 1)
         message(STATUS "Found BLAS library")
      endif()
      mark_as_advanced( generic_blas)
   endif()




   # Make sure we really found a CBLAS library.  That is, it needs to expose
   # the proper cblas link symbols.  So here we test if one of them is present
   # and assume everything is good if it is. Note that we don't do this check if
   # we found the Intel MKL since for some reason CHECK_FUNCTION_EXISTS doesn't work
   # with it.  But it's fine since the MKL should always have cblas.
   if (blas_found AND NOT found_intel_mkl)
      set(CMAKE_REQUIRED_LIBRARIES ${blas_libraries})
      CHECK_FUNCTION_EXISTS(cblas_ddot HAVE_CBLAS)
      if (NOT HAVE_CBLAS)
         message(STATUS "BLAS library does not have cblas symbols, so dlib will not use BLAS or LAPACK")
         set(blas_found 0)
         set(lapack_found 0)
      endif()
   endif()



elseif(WIN32 AND NOT MINGW)
   message(STATUS "Searching for BLAS and LAPACK")

   include(CheckTypeSize)
   check_type_size( "void*" SIZE_OF_VOID_PTR)
   if (SIZE_OF_VOID_PTR EQUAL 8)
      set( mkl_search_path
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/lib/intel64" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/tbb/lib/intel64/vc14" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/compiler/lib/intel64" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64"
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/vc14"
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/lib/intel64" 
         "C:/Program Files (x86)/Intel/Composer XE/mkl/lib/intel64"
         "C:/Program Files (x86)/Intel/Composer XE/tbb/lib/intel64/vc14"
         "C:/Program Files (x86)/Intel/Composer XE/compiler/lib/intel64"
         "C:/Program Files/Intel/Composer XE/mkl/lib/intel64"
         "C:/Program Files/Intel/Composer XE/tbb/lib/intel64/vc14"
         "C:/Program Files/Intel/Composer XE/compiler/lib/intel64"
         )
      find_library(mkl_intel  mkl_intel_lp64 ${mkl_search_path})
   else()
      set( mkl_search_path
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/lib/ia32" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/tbb/lib/ia32/vc14" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/compiler/lib/ia32"
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/ia32" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/ia32/vc14" 
         "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/lib/ia32"
         "C:/Program Files (x86)/Intel/Composer XE/mkl/lib/ia32"
         "C:/Program Files (x86)/Intel/Composer XE/tbb/lib/ia32/vc14"
         "C:/Program Files (x86)/Intel/Composer XE/compiler/lib/ia32"
         "C:/Program Files/Intel/Composer XE/mkl/lib/ia32"
         "C:/Program Files/Intel/Composer XE/tbb/lib/ia32/vc14"
         "C:/Program Files/Intel/Composer XE/compiler/lib/ia32"
         )
      find_library(mkl_intel  mkl_intel_c ${mkl_search_path})
   endif()

   INCLUDE (CheckFunctionExists)

   # Get mkl_include_dir
   set(mkl_include_search_path
      "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/mkl/include"
      "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_*/windows/compiler/include"
      "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include"
      "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/compiler/include"
      "C:/Program Files (x86)/Intel/Composer XE/mkl/include"
      "C:/Program Files (x86)/Intel/Composer XE/compiler/include"
      "C:/Program Files/Intel/Composer XE/mkl/include"
      "C:/Program Files/Intel/Composer XE/compiler/include"
      )
   find_path(mkl_include_dir mkl_version.h ${mkl_include_search_path})
   mark_as_advanced(mkl_include_dir)

   # Search for the needed libraries from the MKL.  
   find_library(mkl_core mkl_core ${mkl_search_path})
   set(mkl_libs ${mkl_intel} ${mkl_core})
   mark_as_advanced(mkl_libs mkl_intel mkl_core)
   if (DLIB_USE_MKL_WITH_TBB)
      find_library(mkl_tbb_thread mkl_tbb_thread ${mkl_search_path})
      find_library(mkl_tbb tbb ${mkl_search_path})
      mark_as_advanced(mkl_tbb_thread mkl_tbb)
      list(APPEND mkl_libs ${mkl_tbb_thread} ${mkl_tbb})
   elseif (DLIB_USE_MKL_SEQUENTIAL)
      find_library(mkl_sequential mkl_sequential ${mkl_search_path})
      mark_as_advanced(mkl_sequential)
      list(APPEND mkl_libs ${mkl_sequential})
   else()
     find_library(mkl_thread mkl_intel_thread ${mkl_search_path})
     find_library(mkl_iomp libiomp5md ${mkl_search_path})
     mark_as_advanced(mkl_thread mkl_iomp)
     list(APPEND mkl_libs ${mkl_thread} ${mkl_iomp})
   endif()

   # If we found the MKL 
   if (mkl_intel AND mkl_core AND ((mkl_tbb_thread AND mkl_tbb) OR mkl_sequential OR (mkl_thread AND mkl_iomp)))
      set(blas_libraries ${mkl_libs})
      set(lapack_libraries ${mkl_libs})
      set(blas_found 1)
      set(lapack_found 1)
      set(found_intel_mkl 1)
      message(STATUS "Found Intel MKL BLAS/LAPACK library")

      # Make sure the version of the Intel MKL we found is compatible with
      # the compiler we are using.  One way to do this check is to see if we can
      # link to it right now.
      set(CMAKE_REQUIRED_LIBRARIES ${blas_libraries})
      CHECK_FUNCTION_EXISTS(cblas_ddot HAVE_CBLAS)
      if (NOT HAVE_CBLAS)
         message("BLAS library does not have cblas symbols, so dlib will not use BLAS or LAPACK")
         set(blas_found 0)
         set(lapack_found 0)
      endif()
   endif()

   if (found_intel_mkl AND mkl_include_dir)
      set(found_intel_mkl_headers 1)
   endif()

endif()


# When all else fails use CMake's built in functions to find BLAS and LAPACK
if (NOT blas_found)
   find_package(BLAS QUIET)
   if (${BLAS_FOUND})
      set(blas_libraries ${BLAS_LIBRARIES})      
      set(blas_found 1)
      if (NOT lapack_found)
         find_package(LAPACK QUIET)
         if (${LAPACK_FOUND})
            set(lapack_libraries ${LAPACK_LIBRARIES})
            set(lapack_found 1)
         endif()
      endif()
   endif()
endif()


# If using lapack, determine whether to mangle functions
if (lapack_found)
   include(CheckFunctionExists)
   include(CheckFortranFunctionExists)
   set(CMAKE_REQUIRED_LIBRARIES ${lapack_libraries})

   check_function_exists("sgesv" LAPACK_FOUND_C_UNMANGLED)
   check_function_exists("sgesv_" LAPACK_FOUND_C_MANGLED)
   if (CMAKE_Fortran_COMPILER_LOADED)
      check_fortran_function_exists("sgesv" LAPACK_FOUND_FORTRAN_UNMANGLED)
      check_fortran_function_exists("sgesv_" LAPACK_FOUND_FORTRAN_MANGLED)
   endif ()
   if (LAPACK_FOUND_C_MANGLED OR LAPACK_FOUND_FORTRAN_MANGLED)
      set(lapack_with_underscore 1)
   elseif (LAPACK_FOUND_C_UNMANGLED OR LAPACK_FOUND_FORTRAN_UNMANGLED)
      set(lapack_without_underscore 1)
   endif ()
endif()


if (UNIX OR MINGW)
   if (NOT blas_found)
      message(" *****************************************************************************")
      message(" *** No BLAS library found so using dlib's built in BLAS.  However, if you ***")
      message(" *** install an optimized BLAS such as OpenBLAS or the Intel MKL your code ***")
      message(" *** will run faster.  On Ubuntu you can install OpenBLAS by executing:    ***")
      message(" ***    sudo apt-get install libopenblas-dev liblapack-dev                 ***")
      message(" *** Or you can easily install OpenBLAS from source by downloading the     ***")
      message(" *** source tar file from http://www.openblas.net, extracting it, and      ***")
      message(" *** running:                                                              ***")
      message(" ***    make; sudo make install                                            ***")
      message(" *****************************************************************************")
   endif()
endif()
