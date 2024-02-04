include(FindPackageHandleStandardArgs)

find_path(CUDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  PATH_SUFFIXES cuda/include cuda include)

find_library(CUDNN_LIBRARY_PATH cudnn
  PATHS ${CUDAToolkit_LIBRARY_DIR}
  PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

add_library(CUDNN UNKNOWN IMPORTED GLOBAL)
add_library(CUDNN::CUDNN ALIAS CUDNN)
set_target_properties(CUDNN PROPERTIES 
    IMPORTED_LOCATION               ${CUDNN_LIBRARY_PATH}
    INTERFACE_INCLUDE_DIRECTORIES   ${CUDNN_INCLUDE_PATH})

mark_as_advanced(CUDNN_INCLUDE_PATH CUDNN_LIBRARY_PATH)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY_PATH)