#
#  _______ _    _ _____  _____     _____  _____                 
# |__   __| |  | |_   _|/ ____|   |_   _|/ ____|       /\       
#    | |  | |__| | | | | (___       | | | (___        /  \      
#    | |  |  __  | | |  \___ \      | |  \___ \      / /\ \     
#    | |  | |  | |_| |_ ____) |    _| |_ ____) |    / ____ \    
#    |_|__|_|_ |_|_____|_____/__  |_____|_____/    /_/  _ \_\   
#   |__   __| |  | |__   __/ __ \|  __ \|_   _|   /\   | |      
#      | |  | |  | |  | | | |  | | |__) | | |    /  \  | |      
#      | |  | |  | |  | | | |  | |  _  /  | |   / /\ \ | |      
#      | |  | |__| |  | | | |__| | | \ \ _| |_ / ____ \| |____  
#      |_|   \____/   |_|  \____/|_|  \_\_____/_/    \_\______| 
#
#
#    _____  ______          _____      _______ _    _ ______       
#   |  __ \|  ____|   /\   |  __ \    |__   __| |  | |  ____|      
#   | |__) | |__     /  \  | |  | |      | |  | |__| | |__         
#   |  _  /|  __|   / /\ \ | |  | |      | |  |  __  |  __|        
#   | | \ \| |____ / ____ \| |__| |      | |  | |  | | |____       
#   |_|__\_\______/_/_ __\_\_____/__ _   |_|__|_|_ |_|______|_ _ _ 
#  / ____/ __ \|  \/  |  \/  |  ____| \ | |__   __/ ____| | | | | |
# | |   | |  | | \  / | \  / | |__  |  \| |  | | | (___   | | | | |
# | |   | |  | | |\/| | |\/| |  __| | . ` |  | |  \___ \  | | | | |
# | |___| |__| | |  | | |  | | |____| |\  |  | |  ____) | |_|_|_|_|
#  \_____\____/|_|  |_|_|  |_|______|_| \_|  |_| |_____/  (_|_|_|_)
#                                                                  
#
#
# This is a CMake makefile.  CMake is a tool that helps you build C++ programs.
# You can download CMake from http://www.cmake.org.  This CMakeLists.txt file
# you are reading builds dlib's example programs. 
#


cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(examples)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# it needs is the dlib source code folder and it will take care of everything.
include(../dlib/cmake)


# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  Here we
# are going to compile one of the dlib example programs which has only one .cpp
# file, assignment_learning_ex.cpp.  If your program consisted of multiple .cpp
# files you would simply list them here in the add_executable() statement.  
add_executable(assignment_learning_ex assignment_learning_ex.cpp)
# Finally, you need to tell CMake that this program, assignment_learning_ex,
# depends on dlib.  You do that with this statement: 
target_link_libraries(assignment_learning_ex dlib::dlib)



# To compile this program all you need to do is ask cmake.  You would type
# these commands from within the directory containing this CMakeLists.txt
# file:
#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
#
# The cmake .. command looks in the parent folder for a file named
# CMakeLists.txt, reads it, sets up everything needed to build program.  Also,
# note that CMake can also generate Visual Studio or XCode project files.  So
# if instead you had written:
#   mkdir build
#   cmake .. -G "Visual Studio 14 2015 Win64" ..
#
# You would be able to open the resulting visual studio project and compile and
# edit the example programs within the visual studio IDE.  CMake can generate a
# lot of different types of IDE projects.  Run the cmake -h command to see a list
# of arguments to -G to see what kinds of projects cmake can generate for you.
# It probably includes your favorite IDE in the list.




#################################################################################
#################################################################################
#  A CMakeLists.txt file can compile more than just one program.  So below we
#  tell it to compile the other dlib example programs using pretty much the
#  same CMake commands we used above.
#################################################################################
#################################################################################


# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
endmacro()

# if an example requires GUI, call this macro to check DLIB_NO_GUI_SUPPORT to include or exclude
macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()

# The deep learning toolkit requires a compiler with essentially complete C++11
# support.  However, versions of Visual Studio prior to October 2016 didn't
# provide enough C++11 support to compile the DNN tooling, but were good enough
# to compile the rest of dlib.  
if (NOT USING_OLD_VISUAL_STUDIO_COMPILER)
   add_example(dnn_metric_learning_ex)
   add_gui_example(dnn_face_recognition_ex)
   add_example(dnn_introduction_ex)
   add_example(dnn_introduction2_ex)
   add_example(dnn_inception_ex)
   add_gui_example(dnn_mmod_ex)
   add_gui_example(dnn_mmod_face_detection_ex)
   add_gui_example(random_cropper_ex)
   add_gui_example(dnn_mmod_dog_hipsterizer)
   add_gui_example(dnn_imagenet_ex)
   add_gui_example(dnn_mmod_find_cars_ex)
   add_example(dnn_mmod_train_find_cars_ex)
   if (NOT MSVC)
      # Don't try to compile this program using Visual Studio since it causes the
      # compiler to run out of RAM and to crash.  Maybe someday Visual Studio
      # won't be broken :(
      add_example(dnn_imagenet_train_ex)
      add_example(dnn_metric_learning_on_images_ex)
   endif()
endif()


if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

      add_executable(webcam_face_pose_ex webcam_face_pose_ex.cpp)
      target_link_libraries(webcam_face_pose_ex dlib::dlib ${OpenCV_LIBS} )
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()



#here we apply our macros 
add_gui_example(3d_point_cloud_ex)
add_example(bayes_net_ex)
add_example(bayes_net_from_disk_ex)
add_gui_example(bayes_net_gui_ex)
add_example(bridge_ex)
add_example(bsp_ex)
add_example(compress_stream_ex)
add_example(config_reader_ex)
add_example(custom_trainer_ex)
add_example(dir_nav_ex)
add_example(empirical_kernel_map_ex)
add_gui_example(face_detection_ex)
add_gui_example(face_landmark_detection_ex)
add_gui_example(fhog_ex)
add_gui_example(fhog_object_detector_ex)
add_example(file_to_code_ex)
add_example(graph_labeling_ex)
add_gui_example(gui_api_ex)
add_gui_example(hough_transform_ex)
add_gui_example(image_ex)
add_example(integrate_function_adapt_simp_ex)
add_example(iosockstream_ex)
add_example(kcentroid_ex)
add_example(kkmeans_ex)
add_example(krls_ex)
add_example(krls_filter_ex)
add_example(krr_classification_ex)
add_example(krr_regression_ex)
add_example(learning_to_track_ex)
add_example(least_squares_ex)
add_example(linear_manifold_regularizer_ex)
add_example(logger_custom_output_ex)
add_example(logger_ex)
add_example(logger_ex_2)
add_example(matrix_ex)
add_example(matrix_expressions_ex)
add_example(max_cost_assignment_ex)
add_example(member_function_pointer_ex)
add_example(mlp_ex)
add_example(model_selection_ex)
add_gui_example(mpc_ex)
add_example(multiclass_classification_ex)
add_example(multithreaded_object_ex)
add_gui_example(object_detector_advanced_ex)
add_gui_example(object_detector_ex)
add_gui_example(one_class_classifiers_ex)
add_example(optimization_ex)
add_example(parallel_for_ex)
add_example(pipe_ex)
add_example(pipe_ex_2)
add_example(quantum_computing_ex)
add_example(queue_ex)
add_example(rank_features_ex)
add_example(running_stats_ex)
add_example(rvm_ex)
add_example(rvm_regression_ex)
add_example(sequence_labeler_ex)
add_example(sequence_segmenter_ex)
add_example(server_http_ex)
add_example(server_iostream_ex)
add_example(sockets_ex)
add_example(sockstreambuf_ex)
add_example(std_allocator_ex)
add_gui_example(surf_ex)
add_example(svm_c_ex)
add_example(svm_ex)
add_example(svm_pegasos_ex)
add_example(svm_rank_ex)
add_example(svm_sparse_ex)
add_example(svm_struct_ex)
add_example(svr_ex)
add_example(thread_function_ex)
add_example(thread_pool_ex)
add_example(threaded_object_ex)
add_example(threads_ex)
add_example(timer_ex)
add_gui_example(train_object_detector)
add_example(train_shape_predictor_ex)
add_example(using_custom_kernels_ex)
add_gui_example(video_tracking_ex)
add_example(xml_parser_ex)


if (DLIB_LINK_WITH_SQLITE3)
   add_example(sqlite_ex)
endif()


