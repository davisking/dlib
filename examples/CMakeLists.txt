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


cmake_minimum_required(VERSION 3.17.0)
# Every project needs a name.  We call this the "examples" project.
project(examples)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(../dlib dlib_build)

# Generator expression helpers
set(IS_WIN32      "$<BOOL:${WIN32}>")
set(IS_NOT_WIN32  "$<NOT:${IS_WIN32}>")

macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib)
   target_link_options(${name} PRIVATE $<$<AND:$<CONFIG:RELEASE>,${IS_NOT_WIN32}>:-s>)
endmacro()

add_example(dnn_metric_learning_ex)
add_example(dnn_introduction_ex)
add_example(dnn_introduction2_ex)
add_example(dnn_introduction3_ex)
add_example(dnn_inception_ex)
add_example(dnn_mmod_train_find_cars_ex)
add_example(dnn_imagenet_train_ex)
add_example(dnn_semantic_segmentation_train_ex)
add_example(dnn_instance_segmentation_train_ex)
add_example(dnn_metric_learning_on_images_ex)
add_example(bayes_net_ex)
add_example(bayes_net_from_disk_ex)
add_example(bridge_ex)
add_example(bsp_ex)
add_example(compress_stream_ex)
add_example(config_reader_ex)
add_example(custom_trainer_ex)
add_example(dir_nav_ex)
add_example(empirical_kernel_map_ex)
add_example(file_to_code_ex)
add_example(graph_labeling_ex)
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
add_example(multiclass_classification_ex)
add_example(multithreaded_object_ex)
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
add_example(train_shape_predictor_ex)
add_example(using_custom_kernels_ex)
add_example(xml_parser_ex)
if (DLIB_LINK_WITH_SQLITE3)
   add_example(sqlite_ex)
endif()
if (NOT DLIB_NO_GUI_SUPPORT)
   add_example(bayes_net_gui_ex)
   add_example(dnn_face_recognition_ex)
   add_example(dnn_dcgan_train_ex)
   add_example(dnn_mmod_ex)
   add_example(dnn_mmod_face_detection_ex)
   add_example(dnn_mmod_find_cars_ex)
   add_example(dnn_mmod_find_cars2_ex)
   add_example(dnn_mmod_dog_hipsterizer)
   add_example(dnn_imagenet_ex)
   add_example(dnn_self_supervised_learning_ex)
   add_example(dnn_semantic_segmentation_ex)
   add_example(dnn_instance_segmentation_ex)
   add_example(dnn_yolo_train_ex)
   add_example(face_detection_ex)
   add_example(face_landmark_detection_ex)
   add_example(fhog_ex)
   add_example(fhog_object_detector_ex)
   add_example(gui_api_ex)
   add_example(hough_transform_ex)
   add_example(image_ex)
   add_example(mpc_ex)
   add_example(object_detector_advanced_ex)
   add_example(object_detector_ex)
   add_example(one_class_classifiers_ex)
   add_example(random_cropper_ex)
   add_example(surf_ex)
   add_example(train_object_detector)
   add_example(video_tracking_ex)
   add_example(3d_point_cloud_ex)
endif()
if (DLIB_USE_FFMPEG AND NOT DLIB_NO_GUI_SUPPORT)
   add_example(ffmpeg_webcam_face_pose_ex)
   add_example(ffmpeg_video_demuxing_ex)
   add_example(ffmpeg_video_demuxing2_ex)
   add_example(ffmpeg_video_decoding_ex)
   add_example(ffmpeg_video_decoding2_ex)
   add_example(ffmpeg_info_ex)
   add_example(ffmpeg_screen_grab_ex)
   add_example(ffmpeg_video_encoding_ex)
   add_example(ffmpeg_video_muxing_ex)
   add_example(ffmpeg_rtsp_ex)
   add_example(ffmpeg_microphone_to_file_ex)
   add_example(ffmpeg_file_to_speaker_ex)
endif()

if (NOT DLIB_NO_GUI_SUPPORT)
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      add_example(webcam_face_pose_ex)
      target_include_directories(webcam_face_pose_ex PRIVATE ${OpenCV_INCLUDE_DIRS})
      target_link_libraries(webcam_face_pose_ex ${OpenCV_LIBS} )
   else()
      message(STATUS "OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()