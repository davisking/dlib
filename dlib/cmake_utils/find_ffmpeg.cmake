#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
#
# This cmake file tries to find installed FFMPEG libraries.

# setting this makes CMake allow normal looking if else statements
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

if (UNIX OR MINGW)
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
            message(STATUS "Found FFMPEG/LIBAV via pkg-config")
        endif()
    else()
        message(STATUS "PkgConfig could not be found, FFMPEG won't be available")
        set(FFMPEG_FOUND 0)
    endif()
endif()

if (UNIX OR MINGW)
    if (NOT FFMPEG_FOUND)
        message(" *****************************************************************************")
        message(" *** No FFMPEG/LIBAV libraries found.                                      ***")
        message(" *** On Ubuntu you can install them by executing                           ***")
        message(" ***    sudo apt install libavdevice-dev libavfilter-dev libavformat-dev   ***")
        message(" ***    sudo apt install libavcodec-dev libswresample-dev libswscale-dev   ***")
        message(" ***    sudo apt install libavutil-dev                                     ***")
        message(" *****************************************************************************")
   endif()
endif()
