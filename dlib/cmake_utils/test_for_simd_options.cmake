# Exercise target selection independently of the machine running this check.
cmake_minimum_required(VERSION 3.10)

function(check_simd_options compiler host target expected)
   set(MSVC TRUE)
   set(CMAKE_CXX_COMPILER_ID "${compiler}")
   set(CMAKE_CXX_COMPILER_VERSION 19.0)
   set(CMAKE_SYSTEM_PROCESSOR "${host}")
   set(CMAKE_CXX_COMPILER_ARCHITECTURE_ID "${target}")

   # Supply the results of the unrelated header and pointer-size probes.
   set(HAVE_SYS_TYPES_H TRUE)
   set(HAVE_STDINT_H TRUE)
   set(HAVE_STDDEF_H TRUE)
   set(HAVE_SIZE_OF_VOID_PTR TRUE)
   set(SIZE_OF_VOID_PTR 8)
   set(USE_SSE2_INSTRUCTIONS ON CACHE BOOL "" FORCE)
   set(USE_SSE4_INSTRUCTIONS OFF CACHE BOOL "" FORCE)
   set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "" FORCE)
   set(USE_NEON_INSTRUCTIONS ON CACHE BOOL "" FORCE)

   include("${CMAKE_CURRENT_LIST_DIR}/set_compiler_specific_options.cmake")
   if (NOT DLIB_SIMD_LEVEL STREQUAL expected)
      message(FATAL_ERROR "${compiler} on ${host} targeting ${target}: expected ${expected}, got ${DLIB_SIMD_LEVEL}")
   endif()
   if (expected STREQUAL "NEON" AND
       "${active_compile_opts};${active_preprocessor_switches}" MATCHES "[Ss][Ss][Ee]|[Aa][Vv][Xx]")
      message(FATAL_ERROR "ARM64 target received x86 SIMD options")
   endif()
endfunction()

check_simd_options(MSVC AMD64 ARM64 NEON)
check_simd_options(MSVC ARM64 ARM64 NEON)
check_simd_options(MSVC ARM64 x64 AVX)
check_simd_options(MSVC AMD64 x64 AVX)
check_simd_options(Clang AMD64 ARM64 NEON)
check_simd_options(Clang ARM64 x64 AVX)
