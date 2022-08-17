include(FindPackageHandleStandardArgs)

set(CUDNN_FOUND False)

if (CUDAToolkit_FOUND)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            PATHS ${CUDAToolkit_ROOT} ${CUDAToolkit_INCLUDE_DIRS})

    find_library(CUDNN_LIBRARY cudnn
            PATHS ${CUDAToolkit_ROOT} ${CUDAToolkit_LIBRARY_DIR}
            PATH_SUFFIXES lib64 lib x64)

    if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

        add_library(cudnn SHARED IMPORTED)
        set_target_properties(cudnn PROPERTIES
                IMPORTED_LOCATION ${CUDNN_LIBRARY} # The DLL, .so or .dylib
                INTERFACE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})

        add_library(CUDNN::CUDNN ALIAS cudnn)
        set(CUDNN_FOUND True)
    endif()
endif()