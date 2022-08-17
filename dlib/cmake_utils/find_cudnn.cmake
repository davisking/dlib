set(CUDNN_FOUND False)

if (CUDAToolkit_FOUND)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDAToolkit_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
            PATHS ${CUDAToolkit_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
            PATH_SUFFIXES include)

    find_library(CUDNN_LIBRARY cudnn
            HINTS ${CUDAToolkit_ROOT} ${CUDAToolkit_LIBRARY_DIR}
            PATHS ${CUDAToolkit_ROOT} ${CUDAToolkit_LIBRARY_DIR}
            PATH_SUFFIXES lib64 lib x64)

    if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
#        message(STATUS "CUDNN_INCLUDE_DIR : ${CUDNN_INCLUDE_DIR} ; CUDNN_LIBRARY : ${CUDNN_LIBRARY}")
        mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
        add_library(cudnn SHARED IMPORTED)
        set_target_properties(cudnn PROPERTIES
                IMPORTED_LOCATION ${CUDNN_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})
        add_library(CUDNN::CUDNN ALIAS cudnn)
        set(CUDNN_FOUND True)
    endif()
endif()