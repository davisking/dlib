include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)

if (PkgConfig_FOUND)
    pkg_check_modules(FFMPEG IMPORTED_TARGET
        libavdevice
        libavfilter
        libavformat
        libavcodec
        libswresample
        libswscale
        libavutil
    )
    if(NOT FFMPEG_FOUND)
        message(" *****************************************************************************")
        message(" *** No FFMPEG/LIBAV libraries found.                                      ***")
        message(" *** On Ubuntu you can install them by executing                           ***")
        message(" ***    sudo apt install libavdevice-dev libavfilter-dev libavformat-dev   ***")
        message(" ***    sudo apt install libavcodec-dev libswresample-dev libswscale-dev   ***")
        message(" ***    sudo apt install libavutil-dev                                     ***")
        message(" *****************************************************************************")
    endif()
endif()

find_package_handle_standard_args(FFMPEG DEFAULT_MSG FFMPEG_LIBRARY_DIRS)