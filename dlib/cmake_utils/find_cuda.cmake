find_package(CUDA)

if (CUDA_FOUND)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_INCLUDE_DIRS}
            PATHS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_INCLUDE_DIRS}
            PATH_SUFFIXES include)

    find_library(CUDNN_LIBRARY cudnn
            HINTS ${CUDA_TOOLKIT_ROOT_DIR}
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib64 lib x64)

    find_program(NVCC_PATH nvcc
            HINTS ${CUDA_TOOLKIT_ROOT_DIR}
            PATHS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES bin)

    if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY AND NVCC_PATH)
        message(STATUS "Found CUDA and CUDNN")
        mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY NVCC_PATH)

        list(APPEND CUDA_LIBRARIES      ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDNN_LIBRARY})
        list(APPEND CUDA_INCLUDE_DIRS   ${CUDNN_INCLUDE_DIR})
        set(CMAKE_CUDA_COMPILER         ${NVCC_PATH})

        cuda_select_nvcc_arch_flags(ARCH_FLAGS All)
        list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS}) # in cmake 3.18 onward, we can just set the property CUDA_ARCHITECTURES to All and be done with it
    endif()
endif()