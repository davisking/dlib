set(CMAKE_BUILD_TYPE RelWithDebInfo) #debug symbols
cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set (target_name wxtest)

PROJECT(${target_name})
add_executable(${target_name} 
  src/wxtest.cpp
  )

#THIS ONLY WORKS WHEN WX IS INSTALLED AS A BINARY. TO MAKE PORTABLE FIGURE OUT THAT SHIT LIKE DLIB
find_package(wxWidgets REQUIRED COMPONENTS net core base)
include(${wxWidgets_USE_FILE})
target_link_libraries(${target_name} ${wxWidgets_LIBRARIES})
