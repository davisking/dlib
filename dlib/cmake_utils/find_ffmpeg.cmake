cmake_minimum_required(VERSION 3.8.0)

message(STATUS "Searching for FFMPEG/LIBAV")
find_package(PkgConfig)
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
    if (FFMPEG_FOUND)
        message(STATUS "Found FFMPEG/LIBAV via pkg-config in `${FFMPEG_LIBRARY_DIRS}`")
    else()
        message(" *****************************************************************************")
        message(" *** No FFMPEG/LIBAV libraries found.                                      ***")
        message(" *** On Ubuntu you can install them by executing                           ***")
        message(" ***    sudo apt install libavdevice-dev libavfilter-dev libavformat-dev   ***")
        message(" ***    sudo apt install libavcodec-dev libswresample-dev libswscale-dev   ***")
        message(" ***    sudo apt install libavutil-dev                                     ***")
        message(" *****************************************************************************")
    endif()
else()
    message(STATUS "PkgConfig could not be found, FFMPEG won't be available")
    set(FFMPEG_FOUND 0)
endif()
