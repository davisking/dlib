include(FindPackageHandleStandardArgs)

find_path(WEBP_INCLUDE_DIR NAMES webp/decode.h)
find_library(WEBP_LIBRARY NAMES webp)

add_library(WEBP UNKNOWN IMPORTED GLOBAL)
add_library(WEBP::WEBP ALIAS WEBP)
set_target_properties(WEBP PROPERTIES 
    IMPORTED_LOCATION               ${WEBP_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES   ${WEBP_INCLUDE_DIR})

mark_as_advanced(WEBP_INCLUDE_DIR WEBP_LIBRARY)
find_package_handle_standard_args(WEBP DEFAULT_MSG WEBP_LIBRARY WEBP_INCLUDE_DIR)