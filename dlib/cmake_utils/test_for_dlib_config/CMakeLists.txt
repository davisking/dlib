cmake_minimum_required(VERSION 3.17.0)

project(dlib_config_test LANGUAGES CXX)

find_package(dlib CONFIG REQUIRED)

get_target_property(dlib_config_test_options dlib::dlib INTERFACE_COMPILE_OPTIONS)
find_package(dlib CONFIG REQUIRED)
get_target_property(dlib_config_test_options_after_second_find dlib::dlib INTERFACE_COMPILE_OPTIONS)
if(NOT "${dlib_config_test_options}" STREQUAL "${dlib_config_test_options_after_second_find}")
   message(FATAL_ERROR "Finding the dlib package twice duplicated its compiler options")
endif()

if(MSVC)
   if(NOT ";${dlib_config_test_options};" MATCHES "/bigobj")
      message(FATAL_ERROR "dlib::dlib did not provide /bigobj to an MSVC-style compiler frontend")
   endif()
elseif(";${dlib_config_test_options};" MATCHES "/bigobj")
   message(FATAL_ERROR "dlib::dlib provided /bigobj to a non-MSVC compiler frontend")
endif()

if(DLIB_CONFIG_TEST_EXPECT_AVX)
   if(MSVC)
      if(NOT ";${dlib_config_test_options};" MATCHES "/arch:AVX")
         message(FATAL_ERROR "dlib::dlib did not translate its AVX requirement to /arch:AVX")
      endif()
   elseif(NOT ";${dlib_config_test_options};" MATCHES "-mavx")
      message(FATAL_ERROR "dlib::dlib did not translate its AVX requirement to -mavx")
   endif()
endif()

add_executable(dlib_config_test dlib_config_test.cpp)
target_link_libraries(dlib_config_test PRIVATE dlib::dlib)

if(DLIB_CONFIG_TEST_EXPECT_AVX)
   target_compile_definitions(dlib_config_test PRIVATE DLIB_CONFIG_TEST_EXPECT_AVX)
endif()
