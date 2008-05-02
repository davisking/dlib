# This is the makefile used to build the dlib C++ library's regression test suite
# on Debian Linux using the gcc compiler.

# this is the name of the output executable
TARGET = test 

# these are the compile time flags passed to gcc
CFLAGS = -ggdb  -DDEBUG -DDLIB_NO_GUI_SUPPORT  -I ../..  -Wall

# These are the link time flags passed to gcc
LFLAGS = -lpthread   -lnsl 

# The name of the compiler.  If you only have one version of
# gcc installed then you probably want to change this to just g++ 
CC = nice g++-3.3

####################################################
####################################################
#  Here we list all the cpp files we want to compile

SRC = main.cpp
SRC += tester.cpp
SRC += ../all/source.cpp 

SRC += example.cpp
SRC += example_args.cpp

SRC += array2d.cpp
SRC += array.cpp
SRC += base64.cpp
SRC += bayes_nets.cpp
SRC += bigint.cpp
SRC += binary_search_tree_kernel_1a.cpp
SRC += binary_search_tree_kernel_2a.cpp
SRC += binary_search_tree_mm1.cpp
SRC += binary_search_tree_mm2.cpp
SRC += cmd_line_parser.cpp
SRC += cmd_line_parser_wchar_t.cpp
SRC += compress_stream.cpp
SRC += conditioning_class.cpp
SRC += conditioning_class_c.cpp
SRC += config_reader.cpp
SRC += directed_graph.cpp
SRC += graph.cpp
SRC += geometry.cpp
SRC += entropy_coder.cpp
SRC += entropy_encoder_model.cpp
SRC += hash_map.cpp
SRC += hash_set.cpp
SRC += hash_table.cpp
SRC += image.cpp
SRC += lz77_buffer.cpp
SRC += map.cpp
SRC += matrix.cpp
SRC += md5.cpp
SRC += member_function_pointer.cpp
SRC += metaprogramming.cpp
SRC += multithreaded_object.cpp
SRC += pipe.cpp
SRC += pixel.cpp
SRC += queue.cpp
SRC += rand.cpp
SRC += reference_counter.cpp
SRC += sequence.cpp
SRC += serialize.cpp
SRC += set.cpp
SRC += sliding_buffer.cpp
SRC += smart_pointers.cpp
SRC += sockets.cpp
SRC += sockstreambuf.cpp
SRC += stack.cpp
SRC += static_map.cpp
SRC += static_set.cpp
SRC += string.cpp
SRC += threads.cpp
SRC += timer.cpp
SRC += tokenizer.cpp
SRC += tuple.cpp


####################################################

TMP = $(SRC:.cpp=.o)
OBJ = $(TMP:.c=.o)

$(TARGET): $(OBJ) 
	@echo Linking $@
	@$(CC) $(LFLAGS) $(OBJ) -o $@
	@echo Build Complete

.cpp.o: $<
	@echo Compiling $<
	@$(CC) -c $(CFLAGS) $< -o $@

clean:
	@rm -f $(OBJ) $(TARGET)
	@echo All object files and binaries removed

dep: 
	@echo Running makedepend
	@makedepend -- $(CFLAGS) -- $(SRC) 2> /dev/null 
	@echo Completed makedepend

###############################################################################
##########  Stuff from makedepend                                         #####
##########  type make dep at the command line to rebuild the dependencies #####
##########  Also, DON'T edit the contents of this file beyond this line.  #####
###############################################################################

# DO NOT DELETE

main.o: ../../dlib/cmd_line_parser.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_1.h ../algs.h
main.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
main.o: ../interfaces/enumerable.h ../interfaces/cmd_line_parser_option.h
main.o: ../assert.h ../string.h ../string/string.h ../error.h
main.o: ../string/string_abstract.h ../uintn.h ../enable_if.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_c.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_abstract.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_print_1.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_print_abstract.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_check_1.h
main.o: ../../dlib/cmd_line_parser/cmd_line_parser_check_c.h ../../dlib/map.h
main.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
main.o: ../interfaces/map_pair.h ../interfaces/remover.h ../serialize.h
main.o: ../algs.h ../uintn.h ../interfaces/enumerable.h
main.o: ../interfaces/map_pair.h ../enable_if.h ../memory_manager.h
main.o: ../memory_manager/memory_manager_kernel_1.h
main.o: ../memory_manager/memory_manager_kernel_abstract.h
main.o: ../memory_manager/memory_manager_kernel_2.h
main.o: ../memory_manager/memory_manager_kernel_3.h
main.o: ../memory_manager/memory_manager_kernel_2.h
main.o: ../binary_search_tree/binary_search_tree_kernel_2.h
main.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
main.o: ../../dlib/map/map_kernel_c.h binary_search_tree.h
main.o: ../../dlib/memory_manager_global.h
main.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
main.o: ../memory_manager/memory_manager_kernel_abstract.h
main.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
main.o: ../../dlib/memory_manager.h ../../dlib/memory_manager_stateless.h
main.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
main.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
main.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
main.o: ../threads.h ../threads/threads_kernel.h ../platform.h
main.o: ../threads/posix.h ../threads/threads_kernel_2.h
main.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
main.o: /usr/include/features.h /usr/include/sys/cdefs.h
main.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
main.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
main.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
main.o: /usr/include/time.h /usr/include/bits/sched.h /usr/include/signal.h
main.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
main.o: /usr/include/bits/setjmp.h /usr/include/errno.h
main.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
main.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
main.o: /usr/include/asm-generic/errno.h
main.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
main.o: /usr/include/bits/time.h /usr/include/sys/select.h
main.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
main.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
main.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
main.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
main.o: ../binary_search_tree/binary_search_tree_kernel_1.h
main.o: ../binary_search_tree/binary_search_tree_kernel_2.h
main.o: ../binary_search_tree/binary_search_tree_kernel_c.h
main.o: ../member_function_pointer.h
main.o: ../member_function_pointer/member_function_pointer_kernel_1.h
main.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
main.o: ../member_function_pointer/member_function_pointer_kernel_c.h
main.o: ../queue.h ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
main.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
main.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
main.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
main.o: ../set/set_kernel_c.h ../set/set_compare_1.h
main.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
main.o: ../threads/auto_unlock_extension.h
main.o: ../threads/auto_unlock_extension_abstract.h
main.o: ../threads/create_new_thread_extension.h
main.o: ../threads/create_new_thread_extension_abstract.h
main.o: ../threads/multithreaded_object_extension.h
main.o: ../threads/multithreaded_object_extension_abstract.h
main.o: ../threads/rsignaler_extension.h
main.o: ../threads/rsignaler_extension_abstract.h ../map.h
main.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
main.o: ../threads/threaded_object_extension.h
main.o: ../threads/threaded_object_extension_abstract.h
main.o: ../threads/thread_specific_data_extension.h
main.o: ../threads/thread_specific_data_extension_abstract.h
main.o: ../threads/thread_function_extension.h
main.o: ../threads/thread_function_extension_abstract.h
main.o: ../threads/threaded_object_extension.h
main.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
main.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../misc_api.h
main.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
main.o: ../misc_api/misc_api_kernel_abstract.h
main.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
main.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
main.o: ../smart_pointers/scoped_ptr_abstract.h
main.o: ../smart_pointers/shared_ptr.h
main.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
main.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
main.o: ../../dlib/logger/extra_logger_headers.h
main.o: ../../dlib/logger/logger_kernel_1.h
main.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
main.o: ../config_reader/config_reader_kernel_1.h
main.o: ../config_reader/config_reader_kernel_abstract.h ../tokenizer.h
main.o: ../tokenizer/tokenizer_kernel_1.h
main.o: ../tokenizer/tokenizer_kernel_abstract.h
main.o: ../tokenizer/tokenizer_kernel_c.h
main.o: ../config_reader/config_reader_thread_safe_1.h
main.o: ../config_reader/config_reader_thread_safe_abstract.h
main.o: ../../dlib/assert.h ../../dlib/algs.h ../../dlib/sequence.h
main.o: ../../dlib/sequence/sequence_kernel_1.h
main.o: ../../dlib/sequence/sequence_kernel_abstract.h
main.o: ../../dlib/sequence/sequence_kernel_2.h
main.o: ../../dlib/sequence/sequence_kernel_c.h
main.o: ../../dlib/sequence/sequence_compare_1.h
main.o: ../../dlib/sequence/sequence_compare_abstract.h
main.o: ../../dlib/sequence/sequence_sort_1.h
main.o: ../../dlib/sequence/sequence_sort_abstract.h
main.o: ../../dlib/sequence/sequence_sort_2.h ../../dlib/string.h
tester.o: tester.h ../../dlib/map.h ../../dlib/logger.h
tester.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
tester.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
tester.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
tester.o: /usr/include/pthread.h /usr/include/features.h
tester.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
tester.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
tester.o: /usr/include/sched.h /usr/include/bits/types.h
tester.o: /usr/include/bits/typesizes.h /usr/include/time.h
tester.o: /usr/include/bits/sched.h /usr/include/signal.h
tester.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
tester.o: /usr/include/bits/setjmp.h /usr/include/errno.h
tester.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
tester.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
tester.o: /usr/include/asm-generic/errno.h
tester.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
tester.o: /usr/include/bits/time.h /usr/include/sys/select.h
tester.o: /usr/include/bits/select.h ../algs.h ../platform.h ../assert.h
tester.o: ../error.h ../noncopyable.h ../threads/threads_kernel_shared.h
tester.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
tester.o: ../threads/rmutex_extension.h
tester.o: ../threads/rmutex_extension_abstract.h
tester.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
tester.o: ../binary_search_tree/binary_search_tree_kernel_1.h
tester.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
tester.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
tester.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
tester.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
tester.o: ../binary_search_tree/binary_search_tree_kernel_2.h
tester.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
tester.o: ../../dlib/memory_manager.h ../member_function_pointer.h
tester.o: ../member_function_pointer/member_function_pointer_kernel_1.h
tester.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
tester.o: ../member_function_pointer/member_function_pointer_kernel_c.h
tester.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
tester.o: ../memory_manager/memory_manager_kernel_abstract.h
tester.o: ../memory_manager/memory_manager_kernel_2.h
tester.o: ../memory_manager/memory_manager_kernel_3.h
tester.o: ../memory_manager/memory_manager_kernel_2.h
tester.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
tester.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
tester.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
tester.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
tester.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
tester.o: ../set/set_kernel_c.h binary_search_tree.h
tester.o: ../../dlib/memory_manager_global.h
tester.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
tester.o: ../memory_manager/memory_manager_kernel_abstract.h
tester.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
tester.o: ../../dlib/memory_manager_stateless.h
tester.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
tester.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
tester.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
tester.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
tester.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
tester.o: ../threads/auto_unlock_extension.h
tester.o: ../threads/auto_unlock_extension_abstract.h
tester.o: ../threads/create_new_thread_extension.h
tester.o: ../threads/create_new_thread_extension_abstract.h
tester.o: ../threads/multithreaded_object_extension.h
tester.o: ../threads/multithreaded_object_extension_abstract.h
tester.o: ../threads/rsignaler_extension.h
tester.o: ../threads/rsignaler_extension_abstract.h ../map.h
tester.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
tester.o: ../threads/threaded_object_extension.h
tester.o: ../threads/threaded_object_extension_abstract.h
tester.o: ../threads/thread_specific_data_extension.h
tester.o: ../threads/thread_specific_data_extension_abstract.h
tester.o: ../threads/thread_function_extension.h
tester.o: ../threads/thread_function_extension_abstract.h
tester.o: ../threads/threaded_object_extension.h ../misc_api.h
tester.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
tester.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
tester.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
tester.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
tester.o: ../smart_pointers/scoped_ptr_abstract.h
tester.o: ../smart_pointers/shared_ptr.h
tester.o: ../smart_pointers/shared_ptr_abstract.h
tester.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
tester.o: ../smart_pointers/weak_ptr_abstract.h
tester.o: ../../dlib/logger/extra_logger_headers.h
tester.o: ../../dlib/logger/logger_kernel_1.h
tester.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
tester.o: ../config_reader/config_reader_kernel_1.h
tester.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
tester.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
tester.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
tester.o: ../tokenizer/tokenizer_kernel_1.h
tester.o: ../tokenizer/tokenizer_kernel_abstract.h
tester.o: ../tokenizer/tokenizer_kernel_c.h
tester.o: ../config_reader/config_reader_thread_safe_1.h
tester.o: ../config_reader/config_reader_thread_safe_abstract.h
tester.o: ../../dlib/assert.h ../../dlib/algs.h ../../dlib/threads.h
../all/source.o: ../base64/base64_kernel_1.cpp ../base64/base64_kernel_1.h
../all/source.o: ../algs.h ../platform.h ../assert.h ../error.h
../all/source.o: ../noncopyable.h ../bigint/bigint_kernel_1.cpp
../all/source.o: ../bigint/bigint_kernel_1.h
../all/source.o: ../bigint/bigint_kernel_abstract.h ../serialize.h ../algs.h
../all/source.o: ../uintn.h ../interfaces/enumerable.h
../all/source.o: ../interfaces/map_pair.h ../enable_if.h ../uintn.h
../all/source.o: ../bigint/bigint_kernel_2.cpp ../bigint/bigint_kernel_2.h
../all/source.o: ../bit_stream/bit_stream_kernel_1.cpp
../all/source.o: ../bit_stream/bit_stream_kernel_1.h
../all/source.o: ../bit_stream/bit_stream_kernel_abstract.h
../all/source.o: ../entropy_decoder/entropy_decoder_kernel_1.cpp
../all/source.o: ../entropy_decoder/entropy_decoder_kernel_1.h
../all/source.o: ../entropy_decoder/entropy_decoder_kernel_abstract.h
../all/source.o: ../assert.h ../entropy_decoder/entropy_decoder_kernel_2.cpp
../all/source.o: ../entropy_decoder/entropy_decoder_kernel_2.h
../all/source.o: ../entropy_encoder/entropy_encoder_kernel_1.cpp
../all/source.o: ../entropy_encoder/entropy_encoder_kernel_1.h
../all/source.o: ../entropy_encoder/entropy_encoder_kernel_abstract.h
../all/source.o: ../entropy_encoder/entropy_encoder_kernel_2.cpp
../all/source.o: ../entropy_encoder/entropy_encoder_kernel_2.h
../all/source.o: ../md5/md5_kernel_1.cpp ../md5/md5_kernel_1.h
../all/source.o: ../md5/md5_kernel_abstract.h ../pixel.cpp ../pixel.h
../all/source.o: ../serialize.h ../tokenizer/tokenizer_kernel_1.cpp
../all/source.o: ../tokenizer/tokenizer_kernel_1.h
../all/source.o: ../tokenizer/tokenizer_kernel_abstract.h
../all/source.o: ../sockets/sockets_kernel_1.cpp ../platform.h
../all/source.o: ../dir_nav/dir_nav_kernel_1.cpp
../all/source.o: ../dir_nav/dir_nav_kernel_2.cpp
../all/source.o: ../dir_nav/dir_nav_kernel_2.h
../all/source.o: ../dir_nav/dir_nav_kernel_abstract.h
../all/source.o: /usr/include/sys/types.h /usr/include/features.h
../all/source.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
../all/source.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
../all/source.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
../all/source.o: /usr/include/time.h /usr/include/endian.h
../all/source.o: /usr/include/bits/endian.h /usr/include/sys/select.h
../all/source.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
../all/source.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
../all/source.o: /usr/include/bits/pthreadtypes.h /usr/include/dirent.h
../all/source.o: /usr/include/bits/dirent.h /usr/include/bits/posix1_lim.h
../all/source.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
../all/source.o: /usr/include/libgen.h /usr/include/unistd.h
../all/source.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
../all/source.o: /usr/include/getopt.h /usr/include/sys/stat.h
../all/source.o: /usr/include/bits/stat.h /usr/include/errno.h
../all/source.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
../all/source.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
../all/source.o: /usr/include/asm-generic/errno.h
../all/source.o: /usr/include/asm-generic/errno-base.h /usr/include/stdlib.h
../all/source.o: /usr/include/alloca.h ../linker/linker_kernel_1.cpp
../all/source.o: ../linker/linker_kernel_1.h
../all/source.o: ../linker/linker_kernel_abstract.h ../threads.h
../all/source.o: ../threads/threads_kernel.h ../threads/posix.h
../all/source.o: ../threads/threads_kernel_2.h
../all/source.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
../all/source.o: /usr/include/sched.h /usr/include/bits/sched.h
../all/source.o: /usr/include/signal.h /usr/include/bits/setjmp.h
../all/source.o: /usr/include/sys/time.h ../threads/threads_kernel_shared.h
../all/source.o: ../threads/auto_mutex_extension.h
../all/source.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
../all/source.o: ../threads/rmutex_extension_abstract.h
../all/source.o: ../threads/auto_mutex_extension_abstract.h
../all/source.o: ../binary_search_tree.h
../all/source.o: ../binary_search_tree/binary_search_tree_kernel_1.h
../all/source.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
../all/source.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
../all/source.o: ../interfaces/remover.h
../all/source.o: ../binary_search_tree/binary_search_tree_kernel_2.h
../all/source.o: ../binary_search_tree/binary_search_tree_kernel_c.h
../all/source.o: ../../dlib/memory_manager.h ../member_function_pointer.h
../all/source.o: ../member_function_pointer/member_function_pointer_kernel_1.h
../all/source.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
../all/source.o: ../member_function_pointer/member_function_pointer_kernel_c.h
../all/source.o: ../memory_manager.h
../all/source.o: ../memory_manager/memory_manager_kernel_1.h
../all/source.o: ../memory_manager/memory_manager_kernel_abstract.h
../all/source.o: ../memory_manager/memory_manager_kernel_2.h
../all/source.o: ../memory_manager/memory_manager_kernel_3.h
../all/source.o: ../memory_manager/memory_manager_kernel_2.h
../all/source.o: ../binary_search_tree/binary_search_tree_kernel_2.h
../all/source.o: ../queue.h ../queue/queue_kernel_1.h
../all/source.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
../all/source.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
../all/source.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
../all/source.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
../all/source.o: ../set/set_kernel_c.h binary_search_tree.h
../all/source.o: ../../dlib/memory_manager_global.h
../all/source.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
../all/source.o: ../memory_manager/memory_manager_kernel_abstract.h
../all/source.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
../all/source.o: ../../dlib/memory_manager_stateless.h
../all/source.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
../all/source.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
../all/source.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
../all/source.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
../all/source.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
../all/source.o: ../misc_api.h ../misc_api/posix.h
../all/source.o: ../misc_api/misc_api_kernel_2.h
../all/source.o: ../misc_api/misc_api_kernel_abstract.h
../all/source.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
../all/source.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
../all/source.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
../all/source.o: ../smart_pointers/shared_ptr.h
../all/source.o: ../smart_pointers/shared_ptr_abstract.h
../all/source.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
../all/source.o: ../smart_pointers/weak_ptr_abstract.h
../all/source.o: ../../dlib/logger/extra_logger_headers.h
../all/source.o: ../../dlib/logger/logger_kernel_1.h
../all/source.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
../all/source.o: ../config_reader/config_reader_kernel_1.h
../all/source.o: ../config_reader/config_reader_kernel_abstract.h
../all/source.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
../all/source.o: ../../dlib/map/map_kernel_abstract.h
../all/source.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
../all/source.o: ../tokenizer/tokenizer_kernel_1.h
../all/source.o: ../tokenizer/tokenizer_kernel_c.h
../all/source.o: ../config_reader/config_reader_thread_safe_1.h
../all/source.o: ../config_reader/config_reader_thread_safe_abstract.h
../all/source.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
../all/source.o: ../set/set_compare_abstract.h
../all/source.o: ../threads/auto_mutex_extension.h
../all/source.o: ../threads/auto_unlock_extension.h
../all/source.o: ../threads/auto_unlock_extension_abstract.h
../all/source.o: ../threads/create_new_thread_extension.h
../all/source.o: ../threads/create_new_thread_extension_abstract.h
../all/source.o: ../threads/multithreaded_object_extension.h
../all/source.o: ../threads/multithreaded_object_extension_abstract.h
../all/source.o: ../threads/rsignaler_extension.h
../all/source.o: ../threads/rsignaler_extension_abstract.h
../all/source.o: ../threads/rmutex_extension.h
../all/source.o: ../threads/rsignaler_extension.h
../all/source.o: ../threads/threaded_object_extension.h
../all/source.o: ../threads/threaded_object_extension_abstract.h
../all/source.o: ../threads/thread_specific_data_extension.h
../all/source.o: ../threads/thread_specific_data_extension_abstract.h
../all/source.o: ../threads/thread_function_extension.h
../all/source.o: ../threads/thread_function_extension_abstract.h
../all/source.o: ../threads/threaded_object_extension.h ../sockets.h
../all/source.o: ../sockets/posix.h ../sockets/sockets_kernel_2.h
../all/source.o: ../sockets/sockets_kernel_abstract.h
../all/source.o: /usr/include/sys/socket.h /usr/include/sys/uio.h
../all/source.o: /usr/include/bits/uio.h /usr/include/bits/socket.h
../all/source.o: /usr/include/limits.h /usr/include/bits/posix2_lim.h
../all/source.o: /usr/include/bits/sockaddr.h /usr/include/asm/socket.h
../all/source.o: /usr/include/asm-i386/socket.h /usr/include/asm/sockios.h
../all/source.o: /usr/include/asm-i386/sockios.h /usr/include/arpa/inet.h
../all/source.o: /usr/include/netinet/in.h /usr/include/stdint.h
../all/source.o: /usr/include/bits/wchar.h /usr/include/bits/in.h
../all/source.o: /usr/include/bits/byteswap.h /usr/include/inttypes.h
../all/source.o: /usr/include/netdb.h /usr/include/rpc/netdb.h
../all/source.o: /usr/include/bits/netdb.h /usr/include/sys/param.h
../all/source.o: /usr/include/linux/param.h /usr/include/asm/param.h
../all/source.o: /usr/include/asm-i386/param.h
../all/source.o: ../sockets/sockets_extensions.h
../all/source.o: ../sockets/sockets_extensions_abstract.h
../all/source.o: ../logger/extra_logger_headers.cpp
../all/source.o: ../logger/extra_logger_headers.h
../all/source.o: ../logger/logger_kernel_1.cpp
../all/source.o: ../logger/logger_config_file.cpp
../all/source.o: ../logger/logger_config_file.h ../error.h ../string.h
../all/source.o: ../string/string.h ../string/string_abstract.h
../all/source.o: ../enable_if.h ../misc_api/misc_api_kernel_1.cpp
../all/source.o: ../misc_api/misc_api_kernel_2.cpp
../all/source.o: ../sockets/sockets_extensions.cpp
../all/source.o: ../sockets/sockets_extensions.h ../timer.h
../all/source.o: ../timer/timer_kernel_1.h ../timer/timer_kernel_abstract.h
../all/source.o: ../timer/timer_kernel_2.h ../timeout.h
../all/source.o: ../timeout/timeout_kernel_1.h
../all/source.o: ../timeout/timeout_kernel_abstract.h
../all/source.o: ../sockets/sockets_kernel_2.cpp /usr/include/fcntl.h
../all/source.o: /usr/include/bits/fcntl.h
../all/source.o: ../sockstreambuf/sockstreambuf_kernel_1.cpp
../all/source.o: ../sockstreambuf/sockstreambuf_kernel_1.h
../all/source.o: ../sockstreambuf/sockstreambuf_kernel_abstract.h
../all/source.o: ../sockstreambuf/sockstreambuf_kernel_2.cpp
../all/source.o: ../sockstreambuf/sockstreambuf_kernel_2.h
../all/source.o: ../threads/multithreaded_object_extension.cpp
../all/source.o: ../threads/multithreaded_object_extension.h
../all/source.o: ../threads/threaded_object_extension.cpp
../all/source.o: ../threads/create_new_thread_extension.h
../all/source.o: ../threads/threads_kernel_1.cpp
../all/source.o: ../threads/threads_kernel_2.cpp
../all/source.o: ../threads/threads_kernel_shared.cpp
../all/source.o: ../timer/timer_kernel_2.cpp ../timer/timer_kernel_2.h
example.o: tester.h ../../dlib/map.h ../../dlib/logger.h
example.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
example.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
example.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
example.o: /usr/include/pthread.h /usr/include/features.h
example.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
example.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
example.o: /usr/include/sched.h /usr/include/bits/types.h
example.o: /usr/include/bits/typesizes.h /usr/include/time.h
example.o: /usr/include/bits/sched.h /usr/include/signal.h
example.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
example.o: /usr/include/bits/setjmp.h /usr/include/errno.h
example.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
example.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
example.o: /usr/include/asm-generic/errno.h
example.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
example.o: /usr/include/bits/time.h /usr/include/sys/select.h
example.o: /usr/include/bits/select.h ../algs.h ../platform.h ../assert.h
example.o: ../error.h ../noncopyable.h ../threads/threads_kernel_shared.h
example.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
example.o: ../threads/rmutex_extension.h
example.o: ../threads/rmutex_extension_abstract.h
example.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
example.o: ../binary_search_tree/binary_search_tree_kernel_1.h
example.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
example.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
example.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
example.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
example.o: ../binary_search_tree/binary_search_tree_kernel_2.h
example.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
example.o: ../../dlib/memory_manager.h ../member_function_pointer.h
example.o: ../member_function_pointer/member_function_pointer_kernel_1.h
example.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
example.o: ../member_function_pointer/member_function_pointer_kernel_c.h
example.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
example.o: ../memory_manager/memory_manager_kernel_abstract.h
example.o: ../memory_manager/memory_manager_kernel_2.h
example.o: ../memory_manager/memory_manager_kernel_3.h
example.o: ../memory_manager/memory_manager_kernel_2.h
example.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
example.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
example.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
example.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
example.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
example.o: ../set/set_kernel_c.h binary_search_tree.h
example.o: ../../dlib/memory_manager_global.h
example.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
example.o: ../memory_manager/memory_manager_kernel_abstract.h
example.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
example.o: ../../dlib/memory_manager_stateless.h
example.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
example.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
example.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
example.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
example.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
example.o: ../threads/auto_unlock_extension.h
example.o: ../threads/auto_unlock_extension_abstract.h
example.o: ../threads/create_new_thread_extension.h
example.o: ../threads/create_new_thread_extension_abstract.h
example.o: ../threads/multithreaded_object_extension.h
example.o: ../threads/multithreaded_object_extension_abstract.h
example.o: ../threads/rsignaler_extension.h
example.o: ../threads/rsignaler_extension_abstract.h ../map.h
example.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
example.o: ../threads/threaded_object_extension.h
example.o: ../threads/threaded_object_extension_abstract.h
example.o: ../threads/thread_specific_data_extension.h
example.o: ../threads/thread_specific_data_extension_abstract.h
example.o: ../threads/thread_function_extension.h
example.o: ../threads/thread_function_extension_abstract.h
example.o: ../threads/threaded_object_extension.h ../misc_api.h
example.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
example.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
example.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
example.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
example.o: ../smart_pointers/scoped_ptr_abstract.h
example.o: ../smart_pointers/shared_ptr.h
example.o: ../smart_pointers/shared_ptr_abstract.h
example.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
example.o: ../smart_pointers/weak_ptr_abstract.h
example.o: ../../dlib/logger/extra_logger_headers.h
example.o: ../../dlib/logger/logger_kernel_1.h
example.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
example.o: ../config_reader/config_reader_kernel_1.h
example.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
example.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
example.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
example.o: ../tokenizer/tokenizer_kernel_1.h
example.o: ../tokenizer/tokenizer_kernel_abstract.h
example.o: ../tokenizer/tokenizer_kernel_c.h
example.o: ../config_reader/config_reader_thread_safe_1.h
example.o: ../config_reader/config_reader_thread_safe_abstract.h
example.o: ../../dlib/assert.h ../../dlib/algs.h
example_args.o: tester.h ../../dlib/map.h ../../dlib/logger.h
example_args.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
example_args.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
example_args.o: ../threads/threads_kernel_2.h
example_args.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
example_args.o: /usr/include/features.h /usr/include/sys/cdefs.h
example_args.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
example_args.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
example_args.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
example_args.o: /usr/include/time.h /usr/include/bits/sched.h
example_args.o: /usr/include/signal.h /usr/include/bits/sigset.h
example_args.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
example_args.o: /usr/include/errno.h /usr/include/bits/errno.h
example_args.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
example_args.o: /usr/include/asm-i386/errno.h
example_args.o: /usr/include/asm-generic/errno.h
example_args.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
example_args.o: /usr/include/bits/time.h /usr/include/sys/select.h
example_args.o: /usr/include/bits/select.h ../algs.h ../platform.h
example_args.o: ../assert.h ../error.h ../noncopyable.h
example_args.o: ../threads/threads_kernel_shared.h
example_args.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
example_args.o: ../threads/rmutex_extension.h
example_args.o: ../threads/rmutex_extension_abstract.h
example_args.o: ../threads/auto_mutex_extension_abstract.h
example_args.o: ../binary_search_tree.h
example_args.o: ../binary_search_tree/binary_search_tree_kernel_1.h
example_args.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
example_args.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
example_args.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
example_args.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
example_args.o: ../enable_if.h
example_args.o: ../binary_search_tree/binary_search_tree_kernel_2.h
example_args.o: ../binary_search_tree/binary_search_tree_kernel_c.h
example_args.o: ../assert.h ../../dlib/memory_manager.h
example_args.o: ../member_function_pointer.h
example_args.o: ../member_function_pointer/member_function_pointer_kernel_1.h
example_args.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
example_args.o: ../member_function_pointer/member_function_pointer_kernel_c.h
example_args.o: ../memory_manager.h
example_args.o: ../memory_manager/memory_manager_kernel_1.h
example_args.o: ../memory_manager/memory_manager_kernel_abstract.h
example_args.o: ../memory_manager/memory_manager_kernel_2.h
example_args.o: ../memory_manager/memory_manager_kernel_3.h
example_args.o: ../memory_manager/memory_manager_kernel_2.h
example_args.o: ../binary_search_tree/binary_search_tree_kernel_2.h
example_args.o: ../queue.h ../queue/queue_kernel_1.h
example_args.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
example_args.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
example_args.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
example_args.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
example_args.o: ../set/set_kernel_c.h binary_search_tree.h
example_args.o: ../../dlib/memory_manager_global.h
example_args.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
example_args.o: ../memory_manager/memory_manager_kernel_abstract.h
example_args.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
example_args.o: ../../dlib/memory_manager_stateless.h
example_args.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
example_args.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
example_args.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
example_args.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
example_args.o: ../set/set_compare_abstract.h
example_args.o: ../threads/auto_mutex_extension.h
example_args.o: ../threads/auto_unlock_extension.h
example_args.o: ../threads/auto_unlock_extension_abstract.h
example_args.o: ../threads/create_new_thread_extension.h
example_args.o: ../threads/create_new_thread_extension_abstract.h
example_args.o: ../threads/multithreaded_object_extension.h
example_args.o: ../threads/multithreaded_object_extension_abstract.h
example_args.o: ../threads/rsignaler_extension.h
example_args.o: ../threads/rsignaler_extension_abstract.h ../map.h
example_args.o: ../threads/rmutex_extension.h
example_args.o: ../threads/rsignaler_extension.h
example_args.o: ../threads/threaded_object_extension.h
example_args.o: ../threads/threaded_object_extension_abstract.h
example_args.o: ../threads/thread_specific_data_extension.h
example_args.o: ../threads/thread_specific_data_extension_abstract.h
example_args.o: ../threads/thread_function_extension.h
example_args.o: ../threads/thread_function_extension_abstract.h
example_args.o: ../threads/threaded_object_extension.h ../misc_api.h
example_args.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
example_args.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
example_args.o: ../../dlib/logger/logger_kernel_abstract.h
example_args.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
example_args.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
example_args.o: ../smart_pointers/shared_ptr.h
example_args.o: ../smart_pointers/shared_ptr_abstract.h
example_args.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
example_args.o: ../smart_pointers/weak_ptr_abstract.h
example_args.o: ../../dlib/logger/extra_logger_headers.h
example_args.o: ../../dlib/logger/logger_kernel_1.h
example_args.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
example_args.o: ../config_reader/config_reader_kernel_1.h
example_args.o: ../config_reader/config_reader_kernel_abstract.h
example_args.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
example_args.o: ../../dlib/map/map_kernel_abstract.h
example_args.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
example_args.o: ../tokenizer/tokenizer_kernel_1.h
example_args.o: ../tokenizer/tokenizer_kernel_abstract.h
example_args.o: ../tokenizer/tokenizer_kernel_c.h
example_args.o: ../config_reader/config_reader_thread_safe_1.h
example_args.o: ../config_reader/config_reader_thread_safe_abstract.h
example_args.o: ../../dlib/assert.h ../../dlib/algs.h
array2d.o: ../../dlib/interfaces/enumerable.h ../../dlib/array2d.h
array2d.o: ../../dlib/array2d/array2d_kernel_1.h
array2d.o: ../../dlib/array2d/array2d_kernel_abstract.h ../algs.h
array2d.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
array2d.o: ../interfaces/enumerable.h ../serialize.h ../algs.h ../uintn.h
array2d.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
array2d.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
array2d.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
array2d.o: ../memory_manager/memory_manager_kernel_2.h
array2d.o: ../memory_manager/memory_manager_kernel_3.h
array2d.o: ../memory_manager/memory_manager_kernel_2.h
array2d.o: ../binary_search_tree/binary_search_tree_kernel_2.h
array2d.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
array2d.o: ../interfaces/map_pair.h ../interfaces/remover.h
array2d.o: ../../dlib/array2d/array2d_kernel_c.h ../../dlib/memory_manager.h
array2d.o: tester.h ../../dlib/map.h ../../dlib/logger.h
array2d.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
array2d.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
array2d.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
array2d.o: /usr/include/pthread.h /usr/include/features.h
array2d.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
array2d.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
array2d.o: /usr/include/sched.h /usr/include/bits/types.h
array2d.o: /usr/include/bits/typesizes.h /usr/include/time.h
array2d.o: /usr/include/bits/sched.h /usr/include/signal.h
array2d.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
array2d.o: /usr/include/bits/setjmp.h /usr/include/errno.h
array2d.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
array2d.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
array2d.o: /usr/include/asm-generic/errno.h
array2d.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
array2d.o: /usr/include/bits/time.h /usr/include/sys/select.h
array2d.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
array2d.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
array2d.o: ../threads/rmutex_extension.h
array2d.o: ../threads/rmutex_extension_abstract.h
array2d.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
array2d.o: ../binary_search_tree/binary_search_tree_kernel_1.h
array2d.o: ../binary_search_tree/binary_search_tree_kernel_2.h
array2d.o: ../binary_search_tree/binary_search_tree_kernel_c.h
array2d.o: ../member_function_pointer.h
array2d.o: ../member_function_pointer/member_function_pointer_kernel_1.h
array2d.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
array2d.o: ../member_function_pointer/member_function_pointer_kernel_c.h
array2d.o: ../queue.h ../queue/queue_kernel_1.h
array2d.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
array2d.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
array2d.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
array2d.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
array2d.o: ../set/set_kernel_c.h binary_search_tree.h
array2d.o: ../../dlib/memory_manager_global.h
array2d.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
array2d.o: ../memory_manager/memory_manager_kernel_abstract.h
array2d.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
array2d.o: ../../dlib/memory_manager_stateless.h
array2d.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
array2d.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
array2d.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
array2d.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
array2d.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
array2d.o: ../threads/auto_unlock_extension.h
array2d.o: ../threads/auto_unlock_extension_abstract.h
array2d.o: ../threads/create_new_thread_extension.h
array2d.o: ../threads/create_new_thread_extension_abstract.h
array2d.o: ../threads/multithreaded_object_extension.h
array2d.o: ../threads/multithreaded_object_extension_abstract.h
array2d.o: ../threads/rsignaler_extension.h
array2d.o: ../threads/rsignaler_extension_abstract.h ../map.h
array2d.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
array2d.o: ../threads/threaded_object_extension.h
array2d.o: ../threads/threaded_object_extension_abstract.h
array2d.o: ../threads/thread_specific_data_extension.h
array2d.o: ../threads/thread_specific_data_extension_abstract.h
array2d.o: ../threads/thread_function_extension.h
array2d.o: ../threads/thread_function_extension_abstract.h
array2d.o: ../threads/threaded_object_extension.h ../misc_api.h
array2d.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
array2d.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
array2d.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
array2d.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
array2d.o: ../smart_pointers/scoped_ptr_abstract.h
array2d.o: ../smart_pointers/shared_ptr.h
array2d.o: ../smart_pointers/shared_ptr_abstract.h
array2d.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
array2d.o: ../smart_pointers/weak_ptr_abstract.h
array2d.o: ../../dlib/logger/extra_logger_headers.h
array2d.o: ../../dlib/logger/logger_kernel_1.h
array2d.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
array2d.o: ../config_reader/config_reader_kernel_1.h
array2d.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
array2d.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
array2d.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
array2d.o: ../tokenizer/tokenizer_kernel_1.h
array2d.o: ../tokenizer/tokenizer_kernel_abstract.h
array2d.o: ../tokenizer/tokenizer_kernel_c.h
array2d.o: ../config_reader/config_reader_thread_safe_1.h
array2d.o: ../config_reader/config_reader_thread_safe_abstract.h
array2d.o: ../../dlib/assert.h ../../dlib/algs.h
array.o: ../../dlib/interfaces/enumerable.h ../../dlib/array.h
array.o: ../../dlib/array/array_kernel_1.h
array.o: ../../dlib/array/array_kernel_abstract.h ../interfaces/enumerable.h
array.o: ../algs.h ../platform.h ../assert.h ../error.h ../noncopyable.h
array.o: ../serialize.h ../algs.h ../uintn.h ../interfaces/enumerable.h
array.o: ../interfaces/map_pair.h ../enable_if.h ../memory_manager.h
array.o: ../memory_manager/memory_manager_kernel_1.h
array.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
array.o: ../memory_manager/memory_manager_kernel_2.h
array.o: ../memory_manager/memory_manager_kernel_3.h
array.o: ../memory_manager/memory_manager_kernel_2.h
array.o: ../binary_search_tree/binary_search_tree_kernel_2.h
array.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
array.o: ../interfaces/map_pair.h ../interfaces/remover.h
array.o: ../../dlib/array/array_kernel_2.h ../../dlib/array/array_kernel_c.h
array.o: ../../dlib/array/array_sort_1.h
array.o: ../../dlib/array/array_sort_abstract.h ../sort.h
array.o: ../../dlib/array/array_sort_2.h ../../dlib/array/array_expand_1.h
array.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
array.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../threads.h
array.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
array.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
array.o: /usr/include/pthread.h /usr/include/features.h
array.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
array.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
array.o: /usr/include/sched.h /usr/include/bits/types.h
array.o: /usr/include/bits/typesizes.h /usr/include/time.h
array.o: /usr/include/bits/sched.h /usr/include/signal.h
array.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
array.o: /usr/include/bits/setjmp.h /usr/include/errno.h
array.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
array.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
array.o: /usr/include/asm-generic/errno.h
array.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
array.o: /usr/include/bits/time.h /usr/include/sys/select.h
array.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
array.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
array.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
array.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
array.o: ../binary_search_tree/binary_search_tree_kernel_1.h
array.o: ../binary_search_tree/binary_search_tree_kernel_2.h
array.o: ../binary_search_tree/binary_search_tree_kernel_c.h
array.o: ../member_function_pointer.h
array.o: ../member_function_pointer/member_function_pointer_kernel_1.h
array.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
array.o: ../member_function_pointer/member_function_pointer_kernel_c.h
array.o: ../queue.h ../queue/queue_kernel_1.h
array.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
array.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
array.o: ../queue/queue_sort_abstract.h ../set.h ../set/set_kernel_1.h
array.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
array.o: binary_search_tree.h ../../dlib/memory_manager_global.h
array.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
array.o: ../memory_manager/memory_manager_kernel_abstract.h
array.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
array.o: ../../dlib/memory_manager_stateless.h
array.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
array.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
array.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
array.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
array.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
array.o: ../threads/auto_unlock_extension.h
array.o: ../threads/auto_unlock_extension_abstract.h
array.o: ../threads/create_new_thread_extension.h
array.o: ../threads/create_new_thread_extension_abstract.h
array.o: ../threads/multithreaded_object_extension.h
array.o: ../threads/multithreaded_object_extension_abstract.h
array.o: ../threads/rsignaler_extension.h
array.o: ../threads/rsignaler_extension_abstract.h ../map.h
array.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
array.o: ../threads/threaded_object_extension.h
array.o: ../threads/threaded_object_extension_abstract.h
array.o: ../threads/thread_specific_data_extension.h
array.o: ../threads/thread_specific_data_extension_abstract.h
array.o: ../threads/thread_function_extension.h
array.o: ../threads/thread_function_extension_abstract.h
array.o: ../threads/threaded_object_extension.h ../misc_api.h
array.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
array.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
array.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
array.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
array.o: ../smart_pointers/scoped_ptr_abstract.h
array.o: ../smart_pointers/shared_ptr.h
array.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
array.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
array.o: ../../dlib/logger/extra_logger_headers.h
array.o: ../../dlib/logger/logger_kernel_1.h
array.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
array.o: ../config_reader/config_reader_kernel_1.h
array.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
array.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
array.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
array.o: ../tokenizer/tokenizer_kernel_1.h
array.o: ../tokenizer/tokenizer_kernel_abstract.h
array.o: ../tokenizer/tokenizer_kernel_c.h
array.o: ../config_reader/config_reader_thread_safe_1.h
array.o: ../config_reader/config_reader_thread_safe_abstract.h
array.o: ../../dlib/assert.h ../../dlib/algs.h
base64.o: ../../dlib/base64.h ../../dlib/base64/base64_kernel_1.h ../algs.h
base64.o: ../platform.h ../assert.h ../error.h ../noncopyable.h tester.h
base64.o: ../../dlib/map.h ../../dlib/logger.h
base64.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
base64.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
base64.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
base64.o: /usr/include/pthread.h /usr/include/features.h
base64.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
base64.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
base64.o: /usr/include/sched.h /usr/include/bits/types.h
base64.o: /usr/include/bits/typesizes.h /usr/include/time.h
base64.o: /usr/include/bits/sched.h /usr/include/signal.h
base64.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
base64.o: /usr/include/bits/setjmp.h /usr/include/errno.h
base64.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
base64.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
base64.o: /usr/include/asm-generic/errno.h
base64.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
base64.o: /usr/include/bits/time.h /usr/include/sys/select.h
base64.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
base64.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
base64.o: ../threads/rmutex_extension.h
base64.o: ../threads/rmutex_extension_abstract.h
base64.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
base64.o: ../binary_search_tree/binary_search_tree_kernel_1.h
base64.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
base64.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
base64.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
base64.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
base64.o: ../binary_search_tree/binary_search_tree_kernel_2.h
base64.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
base64.o: ../../dlib/memory_manager.h ../member_function_pointer.h
base64.o: ../member_function_pointer/member_function_pointer_kernel_1.h
base64.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
base64.o: ../member_function_pointer/member_function_pointer_kernel_c.h
base64.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
base64.o: ../memory_manager/memory_manager_kernel_abstract.h
base64.o: ../memory_manager/memory_manager_kernel_2.h
base64.o: ../memory_manager/memory_manager_kernel_3.h
base64.o: ../memory_manager/memory_manager_kernel_2.h
base64.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
base64.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
base64.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
base64.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
base64.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
base64.o: ../set/set_kernel_c.h binary_search_tree.h
base64.o: ../../dlib/memory_manager_global.h
base64.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
base64.o: ../memory_manager/memory_manager_kernel_abstract.h
base64.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
base64.o: ../../dlib/memory_manager_stateless.h
base64.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
base64.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
base64.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
base64.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
base64.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
base64.o: ../threads/auto_unlock_extension.h
base64.o: ../threads/auto_unlock_extension_abstract.h
base64.o: ../threads/create_new_thread_extension.h
base64.o: ../threads/create_new_thread_extension_abstract.h
base64.o: ../threads/multithreaded_object_extension.h
base64.o: ../threads/multithreaded_object_extension_abstract.h
base64.o: ../threads/rsignaler_extension.h
base64.o: ../threads/rsignaler_extension_abstract.h ../map.h
base64.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
base64.o: ../threads/threaded_object_extension.h
base64.o: ../threads/threaded_object_extension_abstract.h
base64.o: ../threads/thread_specific_data_extension.h
base64.o: ../threads/thread_specific_data_extension_abstract.h
base64.o: ../threads/thread_function_extension.h
base64.o: ../threads/thread_function_extension_abstract.h
base64.o: ../threads/threaded_object_extension.h ../misc_api.h
base64.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
base64.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
base64.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
base64.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
base64.o: ../smart_pointers/scoped_ptr_abstract.h
base64.o: ../smart_pointers/shared_ptr.h
base64.o: ../smart_pointers/shared_ptr_abstract.h
base64.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
base64.o: ../smart_pointers/weak_ptr_abstract.h
base64.o: ../../dlib/logger/extra_logger_headers.h
base64.o: ../../dlib/logger/logger_kernel_1.h
base64.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
base64.o: ../config_reader/config_reader_kernel_1.h
base64.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
base64.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
base64.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
base64.o: ../tokenizer/tokenizer_kernel_1.h
base64.o: ../tokenizer/tokenizer_kernel_abstract.h
base64.o: ../tokenizer/tokenizer_kernel_c.h
base64.o: ../config_reader/config_reader_thread_safe_1.h
base64.o: ../config_reader/config_reader_thread_safe_abstract.h
base64.o: ../../dlib/assert.h ../../dlib/algs.h
bayes_nets.o: ../../dlib/graph_utils.h ../../dlib/graph_utils/graph_utils.h
bayes_nets.o: ../algs.h ../platform.h ../assert.h ../error.h ../noncopyable.h
bayes_nets.o: ../../dlib/graph_utils/graph_utils_abstract.h ../is_kind.h
bayes_nets.o: ../enable_if.h ../set.h ../set/set_kernel_1.h
bayes_nets.o: ../set/set_kernel_abstract.h ../interfaces/enumerable.h
bayes_nets.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
bayes_nets.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
bayes_nets.o: ../enable_if.h ../memory_manager.h
bayes_nets.o: ../memory_manager/memory_manager_kernel_1.h
bayes_nets.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
bayes_nets.o: ../memory_manager/memory_manager_kernel_2.h
bayes_nets.o: ../memory_manager/memory_manager_kernel_3.h
bayes_nets.o: ../memory_manager/memory_manager_kernel_2.h
bayes_nets.o: ../binary_search_tree/binary_search_tree_kernel_2.h
bayes_nets.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
bayes_nets.o: ../interfaces/map_pair.h ../set/set_kernel_c.h
bayes_nets.o: binary_search_tree.h ../../dlib/memory_manager_global.h
bayes_nets.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
bayes_nets.o: ../memory_manager/memory_manager_kernel_abstract.h
bayes_nets.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
bayes_nets.o: ../../dlib/memory_manager.h
bayes_nets.o: ../../dlib/memory_manager_stateless.h
bayes_nets.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
bayes_nets.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
bayes_nets.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
bayes_nets.o: ../threads.h ../threads/threads_kernel.h ../platform.h
bayes_nets.o: ../threads/posix.h ../threads/threads_kernel_2.h
bayes_nets.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
bayes_nets.o: /usr/include/features.h /usr/include/sys/cdefs.h
bayes_nets.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
bayes_nets.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
bayes_nets.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
bayes_nets.o: /usr/include/time.h /usr/include/bits/sched.h
bayes_nets.o: /usr/include/signal.h /usr/include/bits/sigset.h
bayes_nets.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
bayes_nets.o: /usr/include/errno.h /usr/include/bits/errno.h
bayes_nets.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
bayes_nets.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
bayes_nets.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
bayes_nets.o: /usr/include/bits/time.h /usr/include/sys/select.h
bayes_nets.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
bayes_nets.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
bayes_nets.o: ../threads/rmutex_extension.h
bayes_nets.o: ../threads/rmutex_extension_abstract.h
bayes_nets.o: ../threads/auto_mutex_extension_abstract.h
bayes_nets.o: ../binary_search_tree.h
bayes_nets.o: ../binary_search_tree/binary_search_tree_kernel_1.h
bayes_nets.o: ../binary_search_tree/binary_search_tree_kernel_2.h
bayes_nets.o: ../binary_search_tree/binary_search_tree_kernel_c.h
bayes_nets.o: ../member_function_pointer.h
bayes_nets.o: ../member_function_pointer/member_function_pointer_kernel_1.h
bayes_nets.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
bayes_nets.o: ../member_function_pointer/member_function_pointer_kernel_c.h
bayes_nets.o: ../queue.h ../queue/queue_kernel_1.h
bayes_nets.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
bayes_nets.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
bayes_nets.o: ../queue/queue_sort_abstract.h ../sort.h
bayes_nets.o: ../threads/auto_mutex_extension.h
bayes_nets.o: ../threads/auto_unlock_extension.h
bayes_nets.o: ../threads/auto_unlock_extension_abstract.h
bayes_nets.o: ../threads/create_new_thread_extension.h
bayes_nets.o: ../threads/create_new_thread_extension_abstract.h
bayes_nets.o: ../threads/multithreaded_object_extension.h
bayes_nets.o: ../threads/multithreaded_object_extension_abstract.h
bayes_nets.o: ../threads/rsignaler_extension.h
bayes_nets.o: ../threads/rsignaler_extension_abstract.h ../map.h
bayes_nets.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
bayes_nets.o: ../threads/threaded_object_extension.h
bayes_nets.o: ../threads/threaded_object_extension_abstract.h
bayes_nets.o: ../threads/thread_specific_data_extension.h
bayes_nets.o: ../threads/thread_specific_data_extension_abstract.h
bayes_nets.o: ../threads/thread_function_extension.h
bayes_nets.o: ../threads/thread_function_extension_abstract.h
bayes_nets.o: ../threads/threaded_object_extension.h
bayes_nets.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
bayes_nets.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
bayes_nets.o: ../misc_api.h ../misc_api/posix.h
bayes_nets.o: ../misc_api/misc_api_kernel_2.h
bayes_nets.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
bayes_nets.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
bayes_nets.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
bayes_nets.o: ../smart_pointers/scoped_ptr_abstract.h
bayes_nets.o: ../smart_pointers/shared_ptr.h
bayes_nets.o: ../smart_pointers/shared_ptr_abstract.h
bayes_nets.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
bayes_nets.o: ../smart_pointers/weak_ptr_abstract.h
bayes_nets.o: ../../dlib/logger/extra_logger_headers.h
bayes_nets.o: ../../dlib/logger/logger_kernel_1.h
bayes_nets.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
bayes_nets.o: ../config_reader/config_reader_kernel_1.h
bayes_nets.o: ../config_reader/config_reader_kernel_abstract.h
bayes_nets.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
bayes_nets.o: ../../dlib/map/map_kernel_abstract.h
bayes_nets.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
bayes_nets.o: ../tokenizer/tokenizer_kernel_1.h
bayes_nets.o: ../tokenizer/tokenizer_kernel_abstract.h
bayes_nets.o: ../tokenizer/tokenizer_kernel_c.h
bayes_nets.o: ../config_reader/config_reader_thread_safe_1.h
bayes_nets.o: ../config_reader/config_reader_thread_safe_abstract.h
bayes_nets.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
bayes_nets.o: ../set/set_compare_abstract.h ../set_utils.h
bayes_nets.o: ../set_utils/set_utils.h ../set_utils/set_utils_abstract.h
bayes_nets.o: ../../dlib/graph.h ../../dlib/graph/graph_kernel_1.h
bayes_nets.o: ../std_allocator.h ../../dlib/graph/graph_kernel_abstract.h
bayes_nets.o: ../../dlib/directed_graph.h
bayes_nets.o: ../../dlib/directed_graph/directed_graph_kernel_1.h
bayes_nets.o: ../../dlib/directed_graph/directed_graph_kernel_abstract.h
bayes_nets.o: ../../dlib/bayes_utils.h ../../dlib/bayes_utils/bayes_utils.h
bayes_nets.o: ../../dlib/bayes_utils/bayes_utils_abstract.h ../string.h
bayes_nets.o: ../string/string.h ../error.h ../string/string_abstract.h
bayes_nets.o: ../matrix.h ../matrix/matrix.h ../matrix/matrix_abstract.h
bayes_nets.o: ../matrix/matrix_utilities.h
bayes_nets.o: ../matrix/matrix_utilities_abstract.h ../matrix/matrix.h
bayes_nets.o: ../pixel.h ../serialize.h ../matrix/matrix_math_functions.h
bayes_nets.o: ../matrix/matrix_utilities.h ../rand.h ../rand/rand_kernel_1.h
bayes_nets.o: ../rand/rand_kernel_abstract.h ../rand/mersenne_twister.h
bayes_nets.o: ../rand/rand_float_1.h ../rand/rand_float_abstract.h ../array.h
bayes_nets.o: ../../dlib/array/array_kernel_1.h
bayes_nets.o: ../../dlib/array/array_kernel_abstract.h
bayes_nets.o: ../../dlib/array/array_kernel_2.h
bayes_nets.o: ../../dlib/array/array_kernel_c.h
bayes_nets.o: ../../dlib/array/array_sort_1.h
bayes_nets.o: ../../dlib/array/array_sort_abstract.h
bayes_nets.o: ../../dlib/array/array_sort_2.h
bayes_nets.o: ../../dlib/array/array_expand_1.h ../graph.h ../../dlib/set.h
bigint.o: ../../dlib/bigint.h ../../dlib/bigint/bigint_kernel_1.h
bigint.o: ../bigint/bigint_kernel_abstract.h ../algs.h ../platform.h
bigint.o: ../assert.h ../error.h ../noncopyable.h ../serialize.h ../algs.h
bigint.o: ../uintn.h ../interfaces/enumerable.h ../interfaces/map_pair.h
bigint.o: ../enable_if.h ../uintn.h ../../dlib/bigint/bigint_kernel_2.h
bigint.o: ../../dlib/bigint/bigint_kernel_c.h ../assert.h tester.h
bigint.o: ../../dlib/map.h ../../dlib/logger.h
bigint.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
bigint.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
bigint.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
bigint.o: /usr/include/pthread.h /usr/include/features.h
bigint.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
bigint.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
bigint.o: /usr/include/sched.h /usr/include/bits/types.h
bigint.o: /usr/include/bits/typesizes.h /usr/include/time.h
bigint.o: /usr/include/bits/sched.h /usr/include/signal.h
bigint.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
bigint.o: /usr/include/bits/setjmp.h /usr/include/errno.h
bigint.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
bigint.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
bigint.o: /usr/include/asm-generic/errno.h
bigint.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
bigint.o: /usr/include/bits/time.h /usr/include/sys/select.h
bigint.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
bigint.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
bigint.o: ../threads/rmutex_extension.h
bigint.o: ../threads/rmutex_extension_abstract.h
bigint.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
bigint.o: ../binary_search_tree/binary_search_tree_kernel_1.h
bigint.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
bigint.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
bigint.o: ../interfaces/remover.h
bigint.o: ../binary_search_tree/binary_search_tree_kernel_2.h
bigint.o: ../binary_search_tree/binary_search_tree_kernel_c.h
bigint.o: ../../dlib/memory_manager.h ../member_function_pointer.h
bigint.o: ../member_function_pointer/member_function_pointer_kernel_1.h
bigint.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
bigint.o: ../member_function_pointer/member_function_pointer_kernel_c.h
bigint.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
bigint.o: ../memory_manager/memory_manager_kernel_abstract.h
bigint.o: ../memory_manager/memory_manager_kernel_2.h
bigint.o: ../memory_manager/memory_manager_kernel_3.h
bigint.o: ../memory_manager/memory_manager_kernel_2.h
bigint.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
bigint.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
bigint.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
bigint.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
bigint.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
bigint.o: ../set/set_kernel_c.h binary_search_tree.h
bigint.o: ../../dlib/memory_manager_global.h
bigint.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
bigint.o: ../memory_manager/memory_manager_kernel_abstract.h
bigint.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
bigint.o: ../../dlib/memory_manager_stateless.h
bigint.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
bigint.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
bigint.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
bigint.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
bigint.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
bigint.o: ../threads/auto_unlock_extension.h
bigint.o: ../threads/auto_unlock_extension_abstract.h
bigint.o: ../threads/create_new_thread_extension.h
bigint.o: ../threads/create_new_thread_extension_abstract.h
bigint.o: ../threads/multithreaded_object_extension.h
bigint.o: ../threads/multithreaded_object_extension_abstract.h
bigint.o: ../threads/rsignaler_extension.h
bigint.o: ../threads/rsignaler_extension_abstract.h ../map.h
bigint.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
bigint.o: ../threads/threaded_object_extension.h
bigint.o: ../threads/threaded_object_extension_abstract.h
bigint.o: ../threads/thread_specific_data_extension.h
bigint.o: ../threads/thread_specific_data_extension_abstract.h
bigint.o: ../threads/thread_function_extension.h
bigint.o: ../threads/thread_function_extension_abstract.h
bigint.o: ../threads/threaded_object_extension.h ../misc_api.h
bigint.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
bigint.o: ../misc_api/misc_api_kernel_abstract.h
bigint.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
bigint.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
bigint.o: ../smart_pointers/scoped_ptr_abstract.h
bigint.o: ../smart_pointers/shared_ptr.h
bigint.o: ../smart_pointers/shared_ptr_abstract.h
bigint.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
bigint.o: ../smart_pointers/weak_ptr_abstract.h
bigint.o: ../../dlib/logger/extra_logger_headers.h
bigint.o: ../../dlib/logger/logger_kernel_1.h
bigint.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
bigint.o: ../config_reader/config_reader_kernel_1.h
bigint.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
bigint.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
bigint.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
bigint.o: ../tokenizer/tokenizer_kernel_1.h
bigint.o: ../tokenizer/tokenizer_kernel_abstract.h
bigint.o: ../tokenizer/tokenizer_kernel_c.h
bigint.o: ../config_reader/config_reader_thread_safe_1.h
bigint.o: ../config_reader/config_reader_thread_safe_abstract.h
bigint.o: ../../dlib/assert.h ../../dlib/algs.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_global.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
binary_search_tree_kernel_1a.o: ../algs.h ../platform.h ../assert.h
binary_search_tree_kernel_1a.o: ../error.h ../noncopyable.h
binary_search_tree_kernel_1a.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_stateless.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
binary_search_tree_kernel_1a.o: ../threads.h ../threads/threads_kernel.h
binary_search_tree_kernel_1a.o: ../platform.h ../threads/posix.h
binary_search_tree_kernel_1a.o: ../threads/threads_kernel_2.h
binary_search_tree_kernel_1a.o: ../threads/threads_kernel_abstract.h
binary_search_tree_kernel_1a.o: /usr/include/pthread.h
binary_search_tree_kernel_1a.o: /usr/include/features.h
binary_search_tree_kernel_1a.o: /usr/include/sys/cdefs.h
binary_search_tree_kernel_1a.o: /usr/include/bits/wordsize.h
binary_search_tree_kernel_1a.o: /usr/include/gnu/stubs.h
binary_search_tree_kernel_1a.o: /usr/include/gnu/stubs-32.h
binary_search_tree_kernel_1a.o: /usr/include/sched.h
binary_search_tree_kernel_1a.o: /usr/include/bits/types.h
binary_search_tree_kernel_1a.o: /usr/include/bits/typesizes.h
binary_search_tree_kernel_1a.o: /usr/include/time.h /usr/include/bits/sched.h
binary_search_tree_kernel_1a.o: /usr/include/signal.h
binary_search_tree_kernel_1a.o: /usr/include/bits/sigset.h
binary_search_tree_kernel_1a.o: /usr/include/bits/pthreadtypes.h
binary_search_tree_kernel_1a.o: /usr/include/bits/setjmp.h
binary_search_tree_kernel_1a.o: /usr/include/errno.h
binary_search_tree_kernel_1a.o: /usr/include/bits/errno.h
binary_search_tree_kernel_1a.o: /usr/include/linux/errno.h
binary_search_tree_kernel_1a.o: /usr/include/asm/errno.h
binary_search_tree_kernel_1a.o: /usr/include/asm-i386/errno.h
binary_search_tree_kernel_1a.o: /usr/include/asm-generic/errno.h
binary_search_tree_kernel_1a.o: /usr/include/asm-generic/errno-base.h
binary_search_tree_kernel_1a.o: /usr/include/sys/time.h
binary_search_tree_kernel_1a.o: /usr/include/bits/time.h
binary_search_tree_kernel_1a.o: /usr/include/sys/select.h
binary_search_tree_kernel_1a.o: /usr/include/bits/select.h
binary_search_tree_kernel_1a.o: ../threads/threads_kernel_shared.h
binary_search_tree_kernel_1a.o: ../threads/auto_mutex_extension.h
binary_search_tree_kernel_1a.o: ../threads/threads_kernel.h
binary_search_tree_kernel_1a.o: ../threads/rmutex_extension.h
binary_search_tree_kernel_1a.o: ../threads/rmutex_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/auto_mutex_extension_abstract.h
binary_search_tree_kernel_1a.o: ../binary_search_tree.h
binary_search_tree_kernel_1a.o: ../binary_search_tree/binary_search_tree_kernel_1.h
binary_search_tree_kernel_1a.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../interfaces/map_pair.h
binary_search_tree_kernel_1a.o: ../interfaces/enumerable.h
binary_search_tree_kernel_1a.o: ../interfaces/remover.h ../serialize.h
binary_search_tree_kernel_1a.o: ../algs.h ../uintn.h
binary_search_tree_kernel_1a.o: ../interfaces/enumerable.h
binary_search_tree_kernel_1a.o: ../interfaces/map_pair.h ../enable_if.h
binary_search_tree_kernel_1a.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_kernel_1a.o: ../binary_search_tree/binary_search_tree_kernel_c.h
binary_search_tree_kernel_1a.o: ../assert.h ../member_function_pointer.h
binary_search_tree_kernel_1a.o: ../member_function_pointer/member_function_pointer_kernel_1.h
binary_search_tree_kernel_1a.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../member_function_pointer/member_function_pointer_kernel_c.h
binary_search_tree_kernel_1a.o: ../memory_manager.h
binary_search_tree_kernel_1a.o: ../memory_manager/memory_manager_kernel_1.h
binary_search_tree_kernel_1a.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_kernel_1a.o: ../memory_manager/memory_manager_kernel_3.h
binary_search_tree_kernel_1a.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_kernel_1a.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_kernel_1a.o: ../queue.h ../queue/queue_kernel_1.h
binary_search_tree_kernel_1a.o: ../queue/queue_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../queue/queue_kernel_2.h
binary_search_tree_kernel_1a.o: ../queue/queue_kernel_c.h
binary_search_tree_kernel_1a.o: ../queue/queue_sort_1.h
binary_search_tree_kernel_1a.o: ../queue/queue_sort_abstract.h ../sort.h
binary_search_tree_kernel_1a.o: ../set.h ../set/set_kernel_1.h
binary_search_tree_kernel_1a.o: ../set/set_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../set/set_kernel_c.h binary_search_tree.h
binary_search_tree_kernel_1a.o: ../../dlib/binary_search_tree.h tester.h
binary_search_tree_kernel_1a.o: ../../dlib/map.h ../../dlib/logger.h
binary_search_tree_kernel_1a.o: ../../dlib/logger/logger_kernel_1.h
binary_search_tree_kernel_1a.o: ../misc_api.h ../misc_api/posix.h
binary_search_tree_kernel_1a.o: ../misc_api/misc_api_kernel_2.h
binary_search_tree_kernel_1a.o: ../misc_api/misc_api_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../uintn.h
binary_search_tree_kernel_1a.o: ../../dlib/logger/logger_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../map.h ../smart_pointers.h
binary_search_tree_kernel_1a.o: ../smart_pointers/scoped_ptr.h
binary_search_tree_kernel_1a.o: ../noncopyable.h
binary_search_tree_kernel_1a.o: ../smart_pointers/scoped_ptr_abstract.h
binary_search_tree_kernel_1a.o: ../smart_pointers/shared_ptr.h
binary_search_tree_kernel_1a.o: ../smart_pointers/shared_ptr_abstract.h
binary_search_tree_kernel_1a.o: ../smart_pointers/weak_ptr.h
binary_search_tree_kernel_1a.o: ../smart_pointers/shared_ptr.h
binary_search_tree_kernel_1a.o: ../smart_pointers/weak_ptr_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/logger/extra_logger_headers.h
binary_search_tree_kernel_1a.o: ../../dlib/logger/logger_kernel_1.h
binary_search_tree_kernel_1a.o: ../../dlib/logger/logger_config_file.h
binary_search_tree_kernel_1a.o: ../config_reader.h
binary_search_tree_kernel_1a.o: ../config_reader/config_reader_kernel_1.h
binary_search_tree_kernel_1a.o: ../config_reader/config_reader_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/map.h
binary_search_tree_kernel_1a.o: ../../dlib/map/map_kernel_1.h
binary_search_tree_kernel_1a.o: ../../dlib/map/map_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
binary_search_tree_kernel_1a.o: ../tokenizer/tokenizer_kernel_1.h
binary_search_tree_kernel_1a.o: ../tokenizer/tokenizer_kernel_abstract.h
binary_search_tree_kernel_1a.o: ../tokenizer/tokenizer_kernel_c.h
binary_search_tree_kernel_1a.o: ../config_reader/config_reader_thread_safe_1.h
binary_search_tree_kernel_1a.o: ../config_reader/config_reader_thread_safe_abstract.h
binary_search_tree_kernel_1a.o: ../../dlib/assert.h ../../dlib/algs.h
binary_search_tree_kernel_1a.o: ../set/set_compare_1.h
binary_search_tree_kernel_1a.o: ../set/set_compare_abstract.h
binary_search_tree_kernel_1a.o: ../threads/auto_mutex_extension.h
binary_search_tree_kernel_1a.o: ../threads/auto_unlock_extension.h
binary_search_tree_kernel_1a.o: ../threads/auto_unlock_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/create_new_thread_extension.h
binary_search_tree_kernel_1a.o: ../threads/create_new_thread_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/multithreaded_object_extension.h
binary_search_tree_kernel_1a.o: ../threads/multithreaded_object_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/rsignaler_extension.h
binary_search_tree_kernel_1a.o: ../threads/rsignaler_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/rmutex_extension.h
binary_search_tree_kernel_1a.o: ../threads/rsignaler_extension.h
binary_search_tree_kernel_1a.o: ../threads/threaded_object_extension.h
binary_search_tree_kernel_1a.o: ../threads/threaded_object_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/thread_specific_data_extension.h
binary_search_tree_kernel_1a.o: ../threads/thread_specific_data_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/thread_function_extension.h
binary_search_tree_kernel_1a.o: ../threads/thread_function_extension_abstract.h
binary_search_tree_kernel_1a.o: ../threads/threaded_object_extension.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_global.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
binary_search_tree_kernel_2a.o: ../algs.h ../platform.h ../assert.h
binary_search_tree_kernel_2a.o: ../error.h ../noncopyable.h
binary_search_tree_kernel_2a.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_stateless.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
binary_search_tree_kernel_2a.o: ../threads.h ../threads/threads_kernel.h
binary_search_tree_kernel_2a.o: ../platform.h ../threads/posix.h
binary_search_tree_kernel_2a.o: ../threads/threads_kernel_2.h
binary_search_tree_kernel_2a.o: ../threads/threads_kernel_abstract.h
binary_search_tree_kernel_2a.o: /usr/include/pthread.h
binary_search_tree_kernel_2a.o: /usr/include/features.h
binary_search_tree_kernel_2a.o: /usr/include/sys/cdefs.h
binary_search_tree_kernel_2a.o: /usr/include/bits/wordsize.h
binary_search_tree_kernel_2a.o: /usr/include/gnu/stubs.h
binary_search_tree_kernel_2a.o: /usr/include/gnu/stubs-32.h
binary_search_tree_kernel_2a.o: /usr/include/sched.h
binary_search_tree_kernel_2a.o: /usr/include/bits/types.h
binary_search_tree_kernel_2a.o: /usr/include/bits/typesizes.h
binary_search_tree_kernel_2a.o: /usr/include/time.h /usr/include/bits/sched.h
binary_search_tree_kernel_2a.o: /usr/include/signal.h
binary_search_tree_kernel_2a.o: /usr/include/bits/sigset.h
binary_search_tree_kernel_2a.o: /usr/include/bits/pthreadtypes.h
binary_search_tree_kernel_2a.o: /usr/include/bits/setjmp.h
binary_search_tree_kernel_2a.o: /usr/include/errno.h
binary_search_tree_kernel_2a.o: /usr/include/bits/errno.h
binary_search_tree_kernel_2a.o: /usr/include/linux/errno.h
binary_search_tree_kernel_2a.o: /usr/include/asm/errno.h
binary_search_tree_kernel_2a.o: /usr/include/asm-i386/errno.h
binary_search_tree_kernel_2a.o: /usr/include/asm-generic/errno.h
binary_search_tree_kernel_2a.o: /usr/include/asm-generic/errno-base.h
binary_search_tree_kernel_2a.o: /usr/include/sys/time.h
binary_search_tree_kernel_2a.o: /usr/include/bits/time.h
binary_search_tree_kernel_2a.o: /usr/include/sys/select.h
binary_search_tree_kernel_2a.o: /usr/include/bits/select.h
binary_search_tree_kernel_2a.o: ../threads/threads_kernel_shared.h
binary_search_tree_kernel_2a.o: ../threads/auto_mutex_extension.h
binary_search_tree_kernel_2a.o: ../threads/threads_kernel.h
binary_search_tree_kernel_2a.o: ../threads/rmutex_extension.h
binary_search_tree_kernel_2a.o: ../threads/rmutex_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/auto_mutex_extension_abstract.h
binary_search_tree_kernel_2a.o: ../binary_search_tree.h
binary_search_tree_kernel_2a.o: ../binary_search_tree/binary_search_tree_kernel_1.h
binary_search_tree_kernel_2a.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../interfaces/map_pair.h
binary_search_tree_kernel_2a.o: ../interfaces/enumerable.h
binary_search_tree_kernel_2a.o: ../interfaces/remover.h ../serialize.h
binary_search_tree_kernel_2a.o: ../algs.h ../uintn.h
binary_search_tree_kernel_2a.o: ../interfaces/enumerable.h
binary_search_tree_kernel_2a.o: ../interfaces/map_pair.h ../enable_if.h
binary_search_tree_kernel_2a.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_kernel_2a.o: ../binary_search_tree/binary_search_tree_kernel_c.h
binary_search_tree_kernel_2a.o: ../assert.h ../member_function_pointer.h
binary_search_tree_kernel_2a.o: ../member_function_pointer/member_function_pointer_kernel_1.h
binary_search_tree_kernel_2a.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../member_function_pointer/member_function_pointer_kernel_c.h
binary_search_tree_kernel_2a.o: ../memory_manager.h
binary_search_tree_kernel_2a.o: ../memory_manager/memory_manager_kernel_1.h
binary_search_tree_kernel_2a.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_kernel_2a.o: ../memory_manager/memory_manager_kernel_3.h
binary_search_tree_kernel_2a.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_kernel_2a.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_kernel_2a.o: ../queue.h ../queue/queue_kernel_1.h
binary_search_tree_kernel_2a.o: ../queue/queue_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../queue/queue_kernel_2.h
binary_search_tree_kernel_2a.o: ../queue/queue_kernel_c.h
binary_search_tree_kernel_2a.o: ../queue/queue_sort_1.h
binary_search_tree_kernel_2a.o: ../queue/queue_sort_abstract.h ../sort.h
binary_search_tree_kernel_2a.o: ../set.h ../set/set_kernel_1.h
binary_search_tree_kernel_2a.o: ../set/set_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../set/set_kernel_c.h binary_search_tree.h
binary_search_tree_kernel_2a.o: ../../dlib/binary_search_tree.h tester.h
binary_search_tree_kernel_2a.o: ../../dlib/map.h ../../dlib/logger.h
binary_search_tree_kernel_2a.o: ../../dlib/logger/logger_kernel_1.h
binary_search_tree_kernel_2a.o: ../misc_api.h ../misc_api/posix.h
binary_search_tree_kernel_2a.o: ../misc_api/misc_api_kernel_2.h
binary_search_tree_kernel_2a.o: ../misc_api/misc_api_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../uintn.h
binary_search_tree_kernel_2a.o: ../../dlib/logger/logger_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../map.h ../smart_pointers.h
binary_search_tree_kernel_2a.o: ../smart_pointers/scoped_ptr.h
binary_search_tree_kernel_2a.o: ../noncopyable.h
binary_search_tree_kernel_2a.o: ../smart_pointers/scoped_ptr_abstract.h
binary_search_tree_kernel_2a.o: ../smart_pointers/shared_ptr.h
binary_search_tree_kernel_2a.o: ../smart_pointers/shared_ptr_abstract.h
binary_search_tree_kernel_2a.o: ../smart_pointers/weak_ptr.h
binary_search_tree_kernel_2a.o: ../smart_pointers/shared_ptr.h
binary_search_tree_kernel_2a.o: ../smart_pointers/weak_ptr_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/logger/extra_logger_headers.h
binary_search_tree_kernel_2a.o: ../../dlib/logger/logger_kernel_1.h
binary_search_tree_kernel_2a.o: ../../dlib/logger/logger_config_file.h
binary_search_tree_kernel_2a.o: ../config_reader.h
binary_search_tree_kernel_2a.o: ../config_reader/config_reader_kernel_1.h
binary_search_tree_kernel_2a.o: ../config_reader/config_reader_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/map.h
binary_search_tree_kernel_2a.o: ../../dlib/map/map_kernel_1.h
binary_search_tree_kernel_2a.o: ../../dlib/map/map_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
binary_search_tree_kernel_2a.o: ../tokenizer/tokenizer_kernel_1.h
binary_search_tree_kernel_2a.o: ../tokenizer/tokenizer_kernel_abstract.h
binary_search_tree_kernel_2a.o: ../tokenizer/tokenizer_kernel_c.h
binary_search_tree_kernel_2a.o: ../config_reader/config_reader_thread_safe_1.h
binary_search_tree_kernel_2a.o: ../config_reader/config_reader_thread_safe_abstract.h
binary_search_tree_kernel_2a.o: ../../dlib/assert.h ../../dlib/algs.h
binary_search_tree_kernel_2a.o: ../set/set_compare_1.h
binary_search_tree_kernel_2a.o: ../set/set_compare_abstract.h
binary_search_tree_kernel_2a.o: ../threads/auto_mutex_extension.h
binary_search_tree_kernel_2a.o: ../threads/auto_unlock_extension.h
binary_search_tree_kernel_2a.o: ../threads/auto_unlock_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/create_new_thread_extension.h
binary_search_tree_kernel_2a.o: ../threads/create_new_thread_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/multithreaded_object_extension.h
binary_search_tree_kernel_2a.o: ../threads/multithreaded_object_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/rsignaler_extension.h
binary_search_tree_kernel_2a.o: ../threads/rsignaler_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/rmutex_extension.h
binary_search_tree_kernel_2a.o: ../threads/rsignaler_extension.h
binary_search_tree_kernel_2a.o: ../threads/threaded_object_extension.h
binary_search_tree_kernel_2a.o: ../threads/threaded_object_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/thread_specific_data_extension.h
binary_search_tree_kernel_2a.o: ../threads/thread_specific_data_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/thread_function_extension.h
binary_search_tree_kernel_2a.o: ../threads/thread_function_extension_abstract.h
binary_search_tree_kernel_2a.o: ../threads/threaded_object_extension.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_global.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
binary_search_tree_mm1.o: ../algs.h ../platform.h ../assert.h ../error.h
binary_search_tree_mm1.o: ../noncopyable.h
binary_search_tree_mm1.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
binary_search_tree_mm1.o: ../../dlib/memory_manager.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_stateless.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
binary_search_tree_mm1.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
binary_search_tree_mm1.o: ../threads.h ../threads/threads_kernel.h
binary_search_tree_mm1.o: ../platform.h ../threads/posix.h
binary_search_tree_mm1.o: ../threads/threads_kernel_2.h
binary_search_tree_mm1.o: ../threads/threads_kernel_abstract.h
binary_search_tree_mm1.o: /usr/include/pthread.h /usr/include/features.h
binary_search_tree_mm1.o: /usr/include/sys/cdefs.h
binary_search_tree_mm1.o: /usr/include/bits/wordsize.h
binary_search_tree_mm1.o: /usr/include/gnu/stubs.h
binary_search_tree_mm1.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
binary_search_tree_mm1.o: /usr/include/bits/types.h
binary_search_tree_mm1.o: /usr/include/bits/typesizes.h /usr/include/time.h
binary_search_tree_mm1.o: /usr/include/bits/sched.h /usr/include/signal.h
binary_search_tree_mm1.o: /usr/include/bits/sigset.h
binary_search_tree_mm1.o: /usr/include/bits/pthreadtypes.h
binary_search_tree_mm1.o: /usr/include/bits/setjmp.h /usr/include/errno.h
binary_search_tree_mm1.o: /usr/include/bits/errno.h
binary_search_tree_mm1.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
binary_search_tree_mm1.o: /usr/include/asm-i386/errno.h
binary_search_tree_mm1.o: /usr/include/asm-generic/errno.h
binary_search_tree_mm1.o: /usr/include/asm-generic/errno-base.h
binary_search_tree_mm1.o: /usr/include/sys/time.h /usr/include/bits/time.h
binary_search_tree_mm1.o: /usr/include/sys/select.h
binary_search_tree_mm1.o: /usr/include/bits/select.h
binary_search_tree_mm1.o: ../threads/threads_kernel_shared.h
binary_search_tree_mm1.o: ../threads/auto_mutex_extension.h
binary_search_tree_mm1.o: ../threads/threads_kernel.h
binary_search_tree_mm1.o: ../threads/rmutex_extension.h
binary_search_tree_mm1.o: ../threads/rmutex_extension_abstract.h
binary_search_tree_mm1.o: ../threads/auto_mutex_extension_abstract.h
binary_search_tree_mm1.o: ../binary_search_tree.h
binary_search_tree_mm1.o: ../binary_search_tree/binary_search_tree_kernel_1.h
binary_search_tree_mm1.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
binary_search_tree_mm1.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
binary_search_tree_mm1.o: ../interfaces/remover.h ../serialize.h ../algs.h
binary_search_tree_mm1.o: ../uintn.h ../interfaces/enumerable.h
binary_search_tree_mm1.o: ../interfaces/map_pair.h ../enable_if.h
binary_search_tree_mm1.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_mm1.o: ../binary_search_tree/binary_search_tree_kernel_c.h
binary_search_tree_mm1.o: ../assert.h ../member_function_pointer.h
binary_search_tree_mm1.o: ../member_function_pointer/member_function_pointer_kernel_1.h
binary_search_tree_mm1.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
binary_search_tree_mm1.o: ../member_function_pointer/member_function_pointer_kernel_c.h
binary_search_tree_mm1.o: ../memory_manager.h
binary_search_tree_mm1.o: ../memory_manager/memory_manager_kernel_1.h
binary_search_tree_mm1.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_mm1.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_mm1.o: ../memory_manager/memory_manager_kernel_3.h
binary_search_tree_mm1.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_mm1.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_mm1.o: ../queue.h ../queue/queue_kernel_1.h
binary_search_tree_mm1.o: ../queue/queue_kernel_abstract.h
binary_search_tree_mm1.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
binary_search_tree_mm1.o: ../queue/queue_sort_1.h
binary_search_tree_mm1.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
binary_search_tree_mm1.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
binary_search_tree_mm1.o: ../set/set_kernel_c.h binary_search_tree.h
binary_search_tree_mm1.o: ../../dlib/binary_search_tree.h tester.h
binary_search_tree_mm1.o: ../../dlib/map.h ../../dlib/logger.h
binary_search_tree_mm1.o: ../../dlib/logger/logger_kernel_1.h ../misc_api.h
binary_search_tree_mm1.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
binary_search_tree_mm1.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
binary_search_tree_mm1.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
binary_search_tree_mm1.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
binary_search_tree_mm1.o: ../noncopyable.h
binary_search_tree_mm1.o: ../smart_pointers/scoped_ptr_abstract.h
binary_search_tree_mm1.o: ../smart_pointers/shared_ptr.h
binary_search_tree_mm1.o: ../smart_pointers/shared_ptr_abstract.h
binary_search_tree_mm1.o: ../smart_pointers/weak_ptr.h
binary_search_tree_mm1.o: ../smart_pointers/shared_ptr.h
binary_search_tree_mm1.o: ../smart_pointers/weak_ptr_abstract.h
binary_search_tree_mm1.o: ../../dlib/logger/extra_logger_headers.h
binary_search_tree_mm1.o: ../../dlib/logger/logger_kernel_1.h
binary_search_tree_mm1.o: ../../dlib/logger/logger_config_file.h
binary_search_tree_mm1.o: ../config_reader.h
binary_search_tree_mm1.o: ../config_reader/config_reader_kernel_1.h
binary_search_tree_mm1.o: ../config_reader/config_reader_kernel_abstract.h
binary_search_tree_mm1.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
binary_search_tree_mm1.o: ../../dlib/map/map_kernel_abstract.h
binary_search_tree_mm1.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
binary_search_tree_mm1.o: ../tokenizer/tokenizer_kernel_1.h
binary_search_tree_mm1.o: ../tokenizer/tokenizer_kernel_abstract.h
binary_search_tree_mm1.o: ../tokenizer/tokenizer_kernel_c.h
binary_search_tree_mm1.o: ../config_reader/config_reader_thread_safe_1.h
binary_search_tree_mm1.o: ../config_reader/config_reader_thread_safe_abstract.h
binary_search_tree_mm1.o: ../../dlib/assert.h ../../dlib/algs.h
binary_search_tree_mm1.o: ../set/set_compare_1.h
binary_search_tree_mm1.o: ../set/set_compare_abstract.h
binary_search_tree_mm1.o: ../threads/auto_mutex_extension.h
binary_search_tree_mm1.o: ../threads/auto_unlock_extension.h
binary_search_tree_mm1.o: ../threads/auto_unlock_extension_abstract.h
binary_search_tree_mm1.o: ../threads/create_new_thread_extension.h
binary_search_tree_mm1.o: ../threads/create_new_thread_extension_abstract.h
binary_search_tree_mm1.o: ../threads/multithreaded_object_extension.h
binary_search_tree_mm1.o: ../threads/multithreaded_object_extension_abstract.h
binary_search_tree_mm1.o: ../threads/rsignaler_extension.h
binary_search_tree_mm1.o: ../threads/rsignaler_extension_abstract.h
binary_search_tree_mm1.o: ../threads/rmutex_extension.h
binary_search_tree_mm1.o: ../threads/rsignaler_extension.h
binary_search_tree_mm1.o: ../threads/threaded_object_extension.h
binary_search_tree_mm1.o: ../threads/threaded_object_extension_abstract.h
binary_search_tree_mm1.o: ../threads/thread_specific_data_extension.h
binary_search_tree_mm1.o: ../threads/thread_specific_data_extension_abstract.h
binary_search_tree_mm1.o: ../threads/thread_function_extension.h
binary_search_tree_mm1.o: ../threads/thread_function_extension_abstract.h
binary_search_tree_mm1.o: ../threads/threaded_object_extension.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_global.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
binary_search_tree_mm2.o: ../algs.h ../platform.h ../assert.h ../error.h
binary_search_tree_mm2.o: ../noncopyable.h
binary_search_tree_mm2.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
binary_search_tree_mm2.o: ../../dlib/memory_manager.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_stateless.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
binary_search_tree_mm2.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
binary_search_tree_mm2.o: ../threads.h ../threads/threads_kernel.h
binary_search_tree_mm2.o: ../platform.h ../threads/posix.h
binary_search_tree_mm2.o: ../threads/threads_kernel_2.h
binary_search_tree_mm2.o: ../threads/threads_kernel_abstract.h
binary_search_tree_mm2.o: /usr/include/pthread.h /usr/include/features.h
binary_search_tree_mm2.o: /usr/include/sys/cdefs.h
binary_search_tree_mm2.o: /usr/include/bits/wordsize.h
binary_search_tree_mm2.o: /usr/include/gnu/stubs.h
binary_search_tree_mm2.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
binary_search_tree_mm2.o: /usr/include/bits/types.h
binary_search_tree_mm2.o: /usr/include/bits/typesizes.h /usr/include/time.h
binary_search_tree_mm2.o: /usr/include/bits/sched.h /usr/include/signal.h
binary_search_tree_mm2.o: /usr/include/bits/sigset.h
binary_search_tree_mm2.o: /usr/include/bits/pthreadtypes.h
binary_search_tree_mm2.o: /usr/include/bits/setjmp.h /usr/include/errno.h
binary_search_tree_mm2.o: /usr/include/bits/errno.h
binary_search_tree_mm2.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
binary_search_tree_mm2.o: /usr/include/asm-i386/errno.h
binary_search_tree_mm2.o: /usr/include/asm-generic/errno.h
binary_search_tree_mm2.o: /usr/include/asm-generic/errno-base.h
binary_search_tree_mm2.o: /usr/include/sys/time.h /usr/include/bits/time.h
binary_search_tree_mm2.o: /usr/include/sys/select.h
binary_search_tree_mm2.o: /usr/include/bits/select.h
binary_search_tree_mm2.o: ../threads/threads_kernel_shared.h
binary_search_tree_mm2.o: ../threads/auto_mutex_extension.h
binary_search_tree_mm2.o: ../threads/threads_kernel.h
binary_search_tree_mm2.o: ../threads/rmutex_extension.h
binary_search_tree_mm2.o: ../threads/rmutex_extension_abstract.h
binary_search_tree_mm2.o: ../threads/auto_mutex_extension_abstract.h
binary_search_tree_mm2.o: ../binary_search_tree.h
binary_search_tree_mm2.o: ../binary_search_tree/binary_search_tree_kernel_1.h
binary_search_tree_mm2.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
binary_search_tree_mm2.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
binary_search_tree_mm2.o: ../interfaces/remover.h ../serialize.h ../algs.h
binary_search_tree_mm2.o: ../uintn.h ../interfaces/enumerable.h
binary_search_tree_mm2.o: ../interfaces/map_pair.h ../enable_if.h
binary_search_tree_mm2.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_mm2.o: ../binary_search_tree/binary_search_tree_kernel_c.h
binary_search_tree_mm2.o: ../assert.h ../member_function_pointer.h
binary_search_tree_mm2.o: ../member_function_pointer/member_function_pointer_kernel_1.h
binary_search_tree_mm2.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
binary_search_tree_mm2.o: ../member_function_pointer/member_function_pointer_kernel_c.h
binary_search_tree_mm2.o: ../memory_manager.h
binary_search_tree_mm2.o: ../memory_manager/memory_manager_kernel_1.h
binary_search_tree_mm2.o: ../memory_manager/memory_manager_kernel_abstract.h
binary_search_tree_mm2.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_mm2.o: ../memory_manager/memory_manager_kernel_3.h
binary_search_tree_mm2.o: ../memory_manager/memory_manager_kernel_2.h
binary_search_tree_mm2.o: ../binary_search_tree/binary_search_tree_kernel_2.h
binary_search_tree_mm2.o: ../queue.h ../queue/queue_kernel_1.h
binary_search_tree_mm2.o: ../queue/queue_kernel_abstract.h
binary_search_tree_mm2.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
binary_search_tree_mm2.o: ../queue/queue_sort_1.h
binary_search_tree_mm2.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
binary_search_tree_mm2.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
binary_search_tree_mm2.o: ../set/set_kernel_c.h binary_search_tree.h
binary_search_tree_mm2.o: ../../dlib/binary_search_tree.h tester.h
binary_search_tree_mm2.o: ../../dlib/map.h ../../dlib/logger.h
binary_search_tree_mm2.o: ../../dlib/logger/logger_kernel_1.h ../misc_api.h
binary_search_tree_mm2.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
binary_search_tree_mm2.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
binary_search_tree_mm2.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
binary_search_tree_mm2.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
binary_search_tree_mm2.o: ../noncopyable.h
binary_search_tree_mm2.o: ../smart_pointers/scoped_ptr_abstract.h
binary_search_tree_mm2.o: ../smart_pointers/shared_ptr.h
binary_search_tree_mm2.o: ../smart_pointers/shared_ptr_abstract.h
binary_search_tree_mm2.o: ../smart_pointers/weak_ptr.h
binary_search_tree_mm2.o: ../smart_pointers/shared_ptr.h
binary_search_tree_mm2.o: ../smart_pointers/weak_ptr_abstract.h
binary_search_tree_mm2.o: ../../dlib/logger/extra_logger_headers.h
binary_search_tree_mm2.o: ../../dlib/logger/logger_kernel_1.h
binary_search_tree_mm2.o: ../../dlib/logger/logger_config_file.h
binary_search_tree_mm2.o: ../config_reader.h
binary_search_tree_mm2.o: ../config_reader/config_reader_kernel_1.h
binary_search_tree_mm2.o: ../config_reader/config_reader_kernel_abstract.h
binary_search_tree_mm2.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
binary_search_tree_mm2.o: ../../dlib/map/map_kernel_abstract.h
binary_search_tree_mm2.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
binary_search_tree_mm2.o: ../tokenizer/tokenizer_kernel_1.h
binary_search_tree_mm2.o: ../tokenizer/tokenizer_kernel_abstract.h
binary_search_tree_mm2.o: ../tokenizer/tokenizer_kernel_c.h
binary_search_tree_mm2.o: ../config_reader/config_reader_thread_safe_1.h
binary_search_tree_mm2.o: ../config_reader/config_reader_thread_safe_abstract.h
binary_search_tree_mm2.o: ../../dlib/assert.h ../../dlib/algs.h
binary_search_tree_mm2.o: ../set/set_compare_1.h
binary_search_tree_mm2.o: ../set/set_compare_abstract.h
binary_search_tree_mm2.o: ../threads/auto_mutex_extension.h
binary_search_tree_mm2.o: ../threads/auto_unlock_extension.h
binary_search_tree_mm2.o: ../threads/auto_unlock_extension_abstract.h
binary_search_tree_mm2.o: ../threads/create_new_thread_extension.h
binary_search_tree_mm2.o: ../threads/create_new_thread_extension_abstract.h
binary_search_tree_mm2.o: ../threads/multithreaded_object_extension.h
binary_search_tree_mm2.o: ../threads/multithreaded_object_extension_abstract.h
binary_search_tree_mm2.o: ../threads/rsignaler_extension.h
binary_search_tree_mm2.o: ../threads/rsignaler_extension_abstract.h
binary_search_tree_mm2.o: ../threads/rmutex_extension.h
binary_search_tree_mm2.o: ../threads/rsignaler_extension.h
binary_search_tree_mm2.o: ../threads/threaded_object_extension.h
binary_search_tree_mm2.o: ../threads/threaded_object_extension_abstract.h
binary_search_tree_mm2.o: ../threads/thread_specific_data_extension.h
binary_search_tree_mm2.o: ../threads/thread_specific_data_extension_abstract.h
binary_search_tree_mm2.o: ../threads/thread_function_extension.h
binary_search_tree_mm2.o: ../threads/thread_function_extension_abstract.h
binary_search_tree_mm2.o: ../threads/threaded_object_extension.h
cmd_line_parser.o: ../../dlib/string.h ../../dlib/cmd_line_parser.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_1.h
cmd_line_parser.o: ../algs.h ../platform.h ../assert.h ../error.h
cmd_line_parser.o: ../noncopyable.h ../interfaces/enumerable.h
cmd_line_parser.o: ../interfaces/cmd_line_parser_option.h ../assert.h
cmd_line_parser.o: ../string.h ../string/string.h ../error.h
cmd_line_parser.o: ../string/string_abstract.h ../uintn.h ../enable_if.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_c.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_abstract.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_print_1.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_print_abstract.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_check_1.h
cmd_line_parser.o: ../../dlib/cmd_line_parser/cmd_line_parser_check_c.h
cmd_line_parser.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
cmd_line_parser.o: ../../dlib/map/map_kernel_abstract.h
cmd_line_parser.o: ../interfaces/map_pair.h ../interfaces/remover.h
cmd_line_parser.o: ../serialize.h ../algs.h ../uintn.h
cmd_line_parser.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
cmd_line_parser.o: ../enable_if.h ../memory_manager.h
cmd_line_parser.o: ../memory_manager/memory_manager_kernel_1.h
cmd_line_parser.o: ../memory_manager/memory_manager_kernel_abstract.h
cmd_line_parser.o: ../memory_manager/memory_manager_kernel_2.h
cmd_line_parser.o: ../memory_manager/memory_manager_kernel_3.h
cmd_line_parser.o: ../memory_manager/memory_manager_kernel_2.h
cmd_line_parser.o: ../binary_search_tree/binary_search_tree_kernel_2.h
cmd_line_parser.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
cmd_line_parser.o: ../../dlib/map/map_kernel_c.h binary_search_tree.h
cmd_line_parser.o: ../../dlib/memory_manager_global.h
cmd_line_parser.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
cmd_line_parser.o: ../memory_manager/memory_manager_kernel_abstract.h
cmd_line_parser.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
cmd_line_parser.o: ../../dlib/memory_manager.h
cmd_line_parser.o: ../../dlib/memory_manager_stateless.h
cmd_line_parser.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
cmd_line_parser.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
cmd_line_parser.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
cmd_line_parser.o: ../threads.h ../threads/threads_kernel.h ../platform.h
cmd_line_parser.o: ../threads/posix.h ../threads/threads_kernel_2.h
cmd_line_parser.o: ../threads/threads_kernel_abstract.h
cmd_line_parser.o: /usr/include/pthread.h /usr/include/features.h
cmd_line_parser.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
cmd_line_parser.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
cmd_line_parser.o: /usr/include/sched.h /usr/include/bits/types.h
cmd_line_parser.o: /usr/include/bits/typesizes.h /usr/include/time.h
cmd_line_parser.o: /usr/include/bits/sched.h /usr/include/signal.h
cmd_line_parser.o: /usr/include/bits/sigset.h
cmd_line_parser.o: /usr/include/bits/pthreadtypes.h
cmd_line_parser.o: /usr/include/bits/setjmp.h /usr/include/errno.h
cmd_line_parser.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
cmd_line_parser.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
cmd_line_parser.o: /usr/include/asm-generic/errno.h
cmd_line_parser.o: /usr/include/asm-generic/errno-base.h
cmd_line_parser.o: /usr/include/sys/time.h /usr/include/bits/time.h
cmd_line_parser.o: /usr/include/sys/select.h /usr/include/bits/select.h
cmd_line_parser.o: ../threads/threads_kernel_shared.h
cmd_line_parser.o: ../threads/auto_mutex_extension.h
cmd_line_parser.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
cmd_line_parser.o: ../threads/rmutex_extension_abstract.h
cmd_line_parser.o: ../threads/auto_mutex_extension_abstract.h
cmd_line_parser.o: ../binary_search_tree.h
cmd_line_parser.o: ../binary_search_tree/binary_search_tree_kernel_1.h
cmd_line_parser.o: ../binary_search_tree/binary_search_tree_kernel_2.h
cmd_line_parser.o: ../binary_search_tree/binary_search_tree_kernel_c.h
cmd_line_parser.o: ../member_function_pointer.h
cmd_line_parser.o: ../member_function_pointer/member_function_pointer_kernel_1.h
cmd_line_parser.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
cmd_line_parser.o: ../member_function_pointer/member_function_pointer_kernel_c.h
cmd_line_parser.o: ../queue.h ../queue/queue_kernel_1.h
cmd_line_parser.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
cmd_line_parser.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
cmd_line_parser.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
cmd_line_parser.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
cmd_line_parser.o: ../set/set_kernel_c.h ../set/set_compare_1.h
cmd_line_parser.o: ../set/set_compare_abstract.h
cmd_line_parser.o: ../threads/auto_mutex_extension.h
cmd_line_parser.o: ../threads/auto_unlock_extension.h
cmd_line_parser.o: ../threads/auto_unlock_extension_abstract.h
cmd_line_parser.o: ../threads/create_new_thread_extension.h
cmd_line_parser.o: ../threads/create_new_thread_extension_abstract.h
cmd_line_parser.o: ../threads/multithreaded_object_extension.h
cmd_line_parser.o: ../threads/multithreaded_object_extension_abstract.h
cmd_line_parser.o: ../threads/rsignaler_extension.h
cmd_line_parser.o: ../threads/rsignaler_extension_abstract.h ../map.h
cmd_line_parser.o: ../threads/rmutex_extension.h
cmd_line_parser.o: ../threads/rsignaler_extension.h
cmd_line_parser.o: ../threads/threaded_object_extension.h
cmd_line_parser.o: ../threads/threaded_object_extension_abstract.h
cmd_line_parser.o: ../threads/thread_specific_data_extension.h
cmd_line_parser.o: ../threads/thread_specific_data_extension_abstract.h
cmd_line_parser.o: ../threads/thread_function_extension.h
cmd_line_parser.o: ../threads/thread_function_extension_abstract.h
cmd_line_parser.o: ../threads/threaded_object_extension.h
cmd_line_parser.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
cmd_line_parser.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
cmd_line_parser.o: ../misc_api.h ../misc_api/posix.h
cmd_line_parser.o: ../misc_api/misc_api_kernel_2.h
cmd_line_parser.o: ../misc_api/misc_api_kernel_abstract.h
cmd_line_parser.o: ../../dlib/logger/logger_kernel_abstract.h
cmd_line_parser.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
cmd_line_parser.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
cmd_line_parser.o: ../smart_pointers/shared_ptr.h
cmd_line_parser.o: ../smart_pointers/shared_ptr_abstract.h
cmd_line_parser.o: ../smart_pointers/weak_ptr.h
cmd_line_parser.o: ../smart_pointers/shared_ptr.h
cmd_line_parser.o: ../smart_pointers/weak_ptr_abstract.h
cmd_line_parser.o: ../../dlib/logger/extra_logger_headers.h
cmd_line_parser.o: ../../dlib/logger/logger_kernel_1.h
cmd_line_parser.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
cmd_line_parser.o: ../config_reader/config_reader_kernel_1.h
cmd_line_parser.o: ../config_reader/config_reader_kernel_abstract.h
cmd_line_parser.o: ../tokenizer.h ../tokenizer/tokenizer_kernel_1.h
cmd_line_parser.o: ../tokenizer/tokenizer_kernel_abstract.h
cmd_line_parser.o: ../tokenizer/tokenizer_kernel_c.h
cmd_line_parser.o: ../config_reader/config_reader_thread_safe_1.h
cmd_line_parser.o: ../config_reader/config_reader_thread_safe_abstract.h
cmd_line_parser.o: ../../dlib/assert.h ../../dlib/algs.h
cmd_line_parser.o: ../../dlib/sequence.h
cmd_line_parser.o: ../../dlib/sequence/sequence_kernel_1.h
cmd_line_parser.o: ../../dlib/sequence/sequence_kernel_abstract.h
cmd_line_parser.o: ../../dlib/sequence/sequence_kernel_2.h
cmd_line_parser.o: ../../dlib/sequence/sequence_kernel_c.h
cmd_line_parser.o: ../../dlib/sequence/sequence_compare_1.h
cmd_line_parser.o: ../../dlib/sequence/sequence_compare_abstract.h
cmd_line_parser.o: ../../dlib/sequence/sequence_sort_1.h
cmd_line_parser.o: ../../dlib/sequence/sequence_sort_abstract.h
cmd_line_parser.o: ../../dlib/sequence/sequence_sort_2.h cmd_line_parser.h
cmd_line_parser_wchar_t.o: ../../dlib/string.h ../../dlib/cmd_line_parser.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_1.h
cmd_line_parser_wchar_t.o: ../algs.h ../platform.h ../assert.h ../error.h
cmd_line_parser_wchar_t.o: ../noncopyable.h ../interfaces/enumerable.h
cmd_line_parser_wchar_t.o: ../interfaces/cmd_line_parser_option.h ../assert.h
cmd_line_parser_wchar_t.o: ../string.h ../string/string.h ../error.h
cmd_line_parser_wchar_t.o: ../string/string_abstract.h ../uintn.h
cmd_line_parser_wchar_t.o: ../enable_if.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_c.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_print_1.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_print_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_check_1.h
cmd_line_parser_wchar_t.o: ../../dlib/cmd_line_parser/cmd_line_parser_check_c.h
cmd_line_parser_wchar_t.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
cmd_line_parser_wchar_t.o: ../../dlib/map/map_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../interfaces/map_pair.h ../interfaces/remover.h
cmd_line_parser_wchar_t.o: ../serialize.h ../algs.h ../uintn.h
cmd_line_parser_wchar_t.o: ../interfaces/enumerable.h
cmd_line_parser_wchar_t.o: ../interfaces/map_pair.h ../enable_if.h
cmd_line_parser_wchar_t.o: ../memory_manager.h
cmd_line_parser_wchar_t.o: ../memory_manager/memory_manager_kernel_1.h
cmd_line_parser_wchar_t.o: ../memory_manager/memory_manager_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../memory_manager/memory_manager_kernel_2.h
cmd_line_parser_wchar_t.o: ../memory_manager/memory_manager_kernel_3.h
cmd_line_parser_wchar_t.o: ../memory_manager/memory_manager_kernel_2.h
cmd_line_parser_wchar_t.o: ../binary_search_tree/binary_search_tree_kernel_2.h
cmd_line_parser_wchar_t.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/map/map_kernel_c.h binary_search_tree.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_global.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
cmd_line_parser_wchar_t.o: ../memory_manager/memory_manager_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_stateless.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
cmd_line_parser_wchar_t.o: ../threads.h ../threads/threads_kernel.h
cmd_line_parser_wchar_t.o: ../platform.h ../threads/posix.h
cmd_line_parser_wchar_t.o: ../threads/threads_kernel_2.h
cmd_line_parser_wchar_t.o: ../threads/threads_kernel_abstract.h
cmd_line_parser_wchar_t.o: /usr/include/pthread.h /usr/include/features.h
cmd_line_parser_wchar_t.o: /usr/include/sys/cdefs.h
cmd_line_parser_wchar_t.o: /usr/include/bits/wordsize.h
cmd_line_parser_wchar_t.o: /usr/include/gnu/stubs.h
cmd_line_parser_wchar_t.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
cmd_line_parser_wchar_t.o: /usr/include/bits/types.h
cmd_line_parser_wchar_t.o: /usr/include/bits/typesizes.h /usr/include/time.h
cmd_line_parser_wchar_t.o: /usr/include/bits/sched.h /usr/include/signal.h
cmd_line_parser_wchar_t.o: /usr/include/bits/sigset.h
cmd_line_parser_wchar_t.o: /usr/include/bits/pthreadtypes.h
cmd_line_parser_wchar_t.o: /usr/include/bits/setjmp.h /usr/include/errno.h
cmd_line_parser_wchar_t.o: /usr/include/bits/errno.h
cmd_line_parser_wchar_t.o: /usr/include/linux/errno.h
cmd_line_parser_wchar_t.o: /usr/include/asm/errno.h
cmd_line_parser_wchar_t.o: /usr/include/asm-i386/errno.h
cmd_line_parser_wchar_t.o: /usr/include/asm-generic/errno.h
cmd_line_parser_wchar_t.o: /usr/include/asm-generic/errno-base.h
cmd_line_parser_wchar_t.o: /usr/include/sys/time.h /usr/include/bits/time.h
cmd_line_parser_wchar_t.o: /usr/include/sys/select.h
cmd_line_parser_wchar_t.o: /usr/include/bits/select.h
cmd_line_parser_wchar_t.o: ../threads/threads_kernel_shared.h
cmd_line_parser_wchar_t.o: ../threads/auto_mutex_extension.h
cmd_line_parser_wchar_t.o: ../threads/threads_kernel.h
cmd_line_parser_wchar_t.o: ../threads/rmutex_extension.h
cmd_line_parser_wchar_t.o: ../threads/rmutex_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/auto_mutex_extension_abstract.h
cmd_line_parser_wchar_t.o: ../binary_search_tree.h
cmd_line_parser_wchar_t.o: ../binary_search_tree/binary_search_tree_kernel_1.h
cmd_line_parser_wchar_t.o: ../binary_search_tree/binary_search_tree_kernel_2.h
cmd_line_parser_wchar_t.o: ../binary_search_tree/binary_search_tree_kernel_c.h
cmd_line_parser_wchar_t.o: ../member_function_pointer.h
cmd_line_parser_wchar_t.o: ../member_function_pointer/member_function_pointer_kernel_1.h
cmd_line_parser_wchar_t.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../member_function_pointer/member_function_pointer_kernel_c.h
cmd_line_parser_wchar_t.o: ../queue.h ../queue/queue_kernel_1.h
cmd_line_parser_wchar_t.o: ../queue/queue_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../queue/queue_kernel_2.h
cmd_line_parser_wchar_t.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
cmd_line_parser_wchar_t.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
cmd_line_parser_wchar_t.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../set/set_kernel_c.h ../set/set_compare_1.h
cmd_line_parser_wchar_t.o: ../set/set_compare_abstract.h
cmd_line_parser_wchar_t.o: ../threads/auto_mutex_extension.h
cmd_line_parser_wchar_t.o: ../threads/auto_unlock_extension.h
cmd_line_parser_wchar_t.o: ../threads/auto_unlock_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/create_new_thread_extension.h
cmd_line_parser_wchar_t.o: ../threads/create_new_thread_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/multithreaded_object_extension.h
cmd_line_parser_wchar_t.o: ../threads/multithreaded_object_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/rsignaler_extension.h
cmd_line_parser_wchar_t.o: ../threads/rsignaler_extension_abstract.h ../map.h
cmd_line_parser_wchar_t.o: ../threads/rmutex_extension.h
cmd_line_parser_wchar_t.o: ../threads/rsignaler_extension.h
cmd_line_parser_wchar_t.o: ../threads/threaded_object_extension.h
cmd_line_parser_wchar_t.o: ../threads/threaded_object_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/thread_specific_data_extension.h
cmd_line_parser_wchar_t.o: ../threads/thread_specific_data_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/thread_function_extension.h
cmd_line_parser_wchar_t.o: ../threads/thread_function_extension_abstract.h
cmd_line_parser_wchar_t.o: ../threads/threaded_object_extension.h
cmd_line_parser_wchar_t.o: ../../dlib/binary_search_tree.h tester.h
cmd_line_parser_wchar_t.o: ../../dlib/map.h ../../dlib/logger.h
cmd_line_parser_wchar_t.o: ../../dlib/logger/logger_kernel_1.h ../misc_api.h
cmd_line_parser_wchar_t.o: ../misc_api/posix.h
cmd_line_parser_wchar_t.o: ../misc_api/misc_api_kernel_2.h
cmd_line_parser_wchar_t.o: ../misc_api/misc_api_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/logger/logger_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
cmd_line_parser_wchar_t.o: ../noncopyable.h
cmd_line_parser_wchar_t.o: ../smart_pointers/scoped_ptr_abstract.h
cmd_line_parser_wchar_t.o: ../smart_pointers/shared_ptr.h
cmd_line_parser_wchar_t.o: ../smart_pointers/shared_ptr_abstract.h
cmd_line_parser_wchar_t.o: ../smart_pointers/weak_ptr.h
cmd_line_parser_wchar_t.o: ../smart_pointers/shared_ptr.h
cmd_line_parser_wchar_t.o: ../smart_pointers/weak_ptr_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/logger/extra_logger_headers.h
cmd_line_parser_wchar_t.o: ../../dlib/logger/logger_kernel_1.h
cmd_line_parser_wchar_t.o: ../../dlib/logger/logger_config_file.h
cmd_line_parser_wchar_t.o: ../config_reader.h
cmd_line_parser_wchar_t.o: ../config_reader/config_reader_kernel_1.h
cmd_line_parser_wchar_t.o: ../config_reader/config_reader_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../tokenizer.h ../tokenizer/tokenizer_kernel_1.h
cmd_line_parser_wchar_t.o: ../tokenizer/tokenizer_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../tokenizer/tokenizer_kernel_c.h
cmd_line_parser_wchar_t.o: ../config_reader/config_reader_thread_safe_1.h
cmd_line_parser_wchar_t.o: ../config_reader/config_reader_thread_safe_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/assert.h ../../dlib/algs.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_kernel_1.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_kernel_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_kernel_2.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_kernel_c.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_compare_1.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_compare_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_sort_1.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_sort_abstract.h
cmd_line_parser_wchar_t.o: ../../dlib/sequence/sequence_sort_2.h
cmd_line_parser_wchar_t.o: cmd_line_parser.h
compress_stream.o: ../../dlib/compress_stream.h
compress_stream.o: ../../dlib/compress_stream/compress_stream_kernel_1.h
compress_stream.o: ../algs.h ../platform.h ../assert.h ../error.h
compress_stream.o: ../noncopyable.h
compress_stream.o: ../../dlib/compress_stream/compress_stream_kernel_abstract.h
compress_stream.o: ../../dlib/compress_stream/compress_stream_kernel_2.h
compress_stream.o: ../../dlib/compress_stream/compress_stream_kernel_3.h
compress_stream.o: ../assert.h conditioning_class.h
compress_stream.o: ../../dlib/conditioning_class.h
compress_stream.o: ../../dlib/conditioning_class/conditioning_class_kernel_1.h
compress_stream.o: ../../dlib/conditioning_class/conditioning_class_kernel_abstract.h
compress_stream.o: ../../dlib/conditioning_class/conditioning_class_kernel_2.h
compress_stream.o: ../../dlib/conditioning_class/conditioning_class_kernel_3.h
compress_stream.o: ../../dlib/conditioning_class/conditioning_class_kernel_4.h
compress_stream.o: ../../dlib/conditioning_class/conditioning_class_kernel_c.h
compress_stream.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
compress_stream.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
compress_stream.o: ../threads.h ../threads/threads_kernel.h ../platform.h
compress_stream.o: ../threads/posix.h ../threads/threads_kernel_2.h
compress_stream.o: ../threads/threads_kernel_abstract.h
compress_stream.o: /usr/include/pthread.h /usr/include/features.h
compress_stream.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
compress_stream.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
compress_stream.o: /usr/include/sched.h /usr/include/bits/types.h
compress_stream.o: /usr/include/bits/typesizes.h /usr/include/time.h
compress_stream.o: /usr/include/bits/sched.h /usr/include/signal.h
compress_stream.o: /usr/include/bits/sigset.h
compress_stream.o: /usr/include/bits/pthreadtypes.h
compress_stream.o: /usr/include/bits/setjmp.h /usr/include/errno.h
compress_stream.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
compress_stream.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
compress_stream.o: /usr/include/asm-generic/errno.h
compress_stream.o: /usr/include/asm-generic/errno-base.h
compress_stream.o: /usr/include/sys/time.h /usr/include/bits/time.h
compress_stream.o: /usr/include/sys/select.h /usr/include/bits/select.h
compress_stream.o: ../threads/threads_kernel_shared.h
compress_stream.o: ../threads/auto_mutex_extension.h
compress_stream.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
compress_stream.o: ../threads/rmutex_extension_abstract.h
compress_stream.o: ../threads/auto_mutex_extension_abstract.h
compress_stream.o: ../binary_search_tree.h
compress_stream.o: ../binary_search_tree/binary_search_tree_kernel_1.h
compress_stream.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
compress_stream.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
compress_stream.o: ../interfaces/remover.h ../serialize.h ../algs.h
compress_stream.o: ../uintn.h ../interfaces/enumerable.h
compress_stream.o: ../interfaces/map_pair.h ../enable_if.h
compress_stream.o: ../binary_search_tree/binary_search_tree_kernel_2.h
compress_stream.o: ../binary_search_tree/binary_search_tree_kernel_c.h
compress_stream.o: ../member_function_pointer.h
compress_stream.o: ../member_function_pointer/member_function_pointer_kernel_1.h
compress_stream.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
compress_stream.o: ../member_function_pointer/member_function_pointer_kernel_c.h
compress_stream.o: ../memory_manager.h
compress_stream.o: ../memory_manager/memory_manager_kernel_1.h
compress_stream.o: ../memory_manager/memory_manager_kernel_abstract.h
compress_stream.o: ../memory_manager/memory_manager_kernel_2.h
compress_stream.o: ../memory_manager/memory_manager_kernel_3.h
compress_stream.o: ../memory_manager/memory_manager_kernel_2.h
compress_stream.o: ../binary_search_tree/binary_search_tree_kernel_2.h
compress_stream.o: ../queue.h ../queue/queue_kernel_1.h
compress_stream.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
compress_stream.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
compress_stream.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
compress_stream.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
compress_stream.o: ../set/set_kernel_c.h binary_search_tree.h
compress_stream.o: ../../dlib/memory_manager_global.h
compress_stream.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
compress_stream.o: ../memory_manager/memory_manager_kernel_abstract.h
compress_stream.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
compress_stream.o: ../../dlib/memory_manager_stateless.h
compress_stream.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
compress_stream.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
compress_stream.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
compress_stream.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
compress_stream.o: ../set/set_compare_abstract.h
compress_stream.o: ../threads/auto_mutex_extension.h
compress_stream.o: ../threads/auto_unlock_extension.h
compress_stream.o: ../threads/auto_unlock_extension_abstract.h
compress_stream.o: ../threads/create_new_thread_extension.h
compress_stream.o: ../threads/create_new_thread_extension_abstract.h
compress_stream.o: ../threads/multithreaded_object_extension.h
compress_stream.o: ../threads/multithreaded_object_extension_abstract.h
compress_stream.o: ../threads/rsignaler_extension.h
compress_stream.o: ../threads/rsignaler_extension_abstract.h ../map.h
compress_stream.o: ../threads/rmutex_extension.h
compress_stream.o: ../threads/rsignaler_extension.h
compress_stream.o: ../threads/threaded_object_extension.h
compress_stream.o: ../threads/threaded_object_extension_abstract.h
compress_stream.o: ../threads/thread_specific_data_extension.h
compress_stream.o: ../threads/thread_specific_data_extension_abstract.h
compress_stream.o: ../threads/thread_function_extension.h
compress_stream.o: ../threads/thread_function_extension_abstract.h
compress_stream.o: ../threads/threaded_object_extension.h ../misc_api.h
compress_stream.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
compress_stream.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
compress_stream.o: ../../dlib/logger/logger_kernel_abstract.h
compress_stream.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
compress_stream.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
compress_stream.o: ../smart_pointers/shared_ptr.h
compress_stream.o: ../smart_pointers/shared_ptr_abstract.h
compress_stream.o: ../smart_pointers/weak_ptr.h
compress_stream.o: ../smart_pointers/shared_ptr.h
compress_stream.o: ../smart_pointers/weak_ptr_abstract.h
compress_stream.o: ../../dlib/logger/extra_logger_headers.h
compress_stream.o: ../../dlib/logger/logger_kernel_1.h
compress_stream.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
compress_stream.o: ../config_reader/config_reader_kernel_1.h
compress_stream.o: ../config_reader/config_reader_kernel_abstract.h
compress_stream.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
compress_stream.o: ../../dlib/map/map_kernel_abstract.h
compress_stream.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
compress_stream.o: ../tokenizer/tokenizer_kernel_1.h
compress_stream.o: ../tokenizer/tokenizer_kernel_abstract.h
compress_stream.o: ../tokenizer/tokenizer_kernel_c.h
compress_stream.o: ../config_reader/config_reader_thread_safe_1.h
compress_stream.o: ../config_reader/config_reader_thread_safe_abstract.h
compress_stream.o: ../../dlib/assert.h ../../dlib/algs.h
compress_stream.o: ../../dlib/entropy_encoder.h
compress_stream.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_1.h
compress_stream.o: ../entropy_encoder/entropy_encoder_kernel_abstract.h
compress_stream.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_2.h
compress_stream.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_c.h
compress_stream.o: ../../dlib/entropy_decoder.h
compress_stream.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_1.h
compress_stream.o: ../entropy_decoder/entropy_decoder_kernel_abstract.h
compress_stream.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_2.h
compress_stream.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_c.h
compress_stream.o: ../../dlib/entropy_encoder_model.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_1.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_abstract.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_2.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_3.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_4.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_5.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_6.h
compress_stream.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_c.h
compress_stream.o: ../../dlib/sliding_buffer.h
compress_stream.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_1.h
compress_stream.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_abstract.h
compress_stream.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_c.h
compress_stream.o: ../../dlib/entropy_decoder_model.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_1.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_abstract.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_2.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_3.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_4.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_5.h
compress_stream.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_6.h
compress_stream.o: ../../dlib/lz77_buffer.h
compress_stream.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_1.h
compress_stream.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_abstract.h
compress_stream.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_2.h
compress_stream.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_c.h
compress_stream.o: ../../dlib/lzp_buffer.h
compress_stream.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_1.h
compress_stream.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_abstract.h
compress_stream.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_2.h
compress_stream.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_c.h
compress_stream.o: ../../dlib/crc32.h ../../dlib/crc32/crc32_kernel_1.h
compress_stream.o: ../../dlib/crc32/crc32_kernel_abstract.h
conditioning_class.o: ../../dlib/conditioning_class.h
conditioning_class.o: ../../dlib/conditioning_class/conditioning_class_kernel_1.h
conditioning_class.o: ../../dlib/conditioning_class/conditioning_class_kernel_abstract.h
conditioning_class.o: ../assert.h ../algs.h ../platform.h ../assert.h
conditioning_class.o: ../error.h ../noncopyable.h
conditioning_class.o: ../../dlib/conditioning_class/conditioning_class_kernel_2.h
conditioning_class.o: ../../dlib/conditioning_class/conditioning_class_kernel_3.h
conditioning_class.o: ../../dlib/conditioning_class/conditioning_class_kernel_4.h
conditioning_class.o: ../../dlib/conditioning_class/conditioning_class_kernel_c.h
conditioning_class.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
conditioning_class.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
conditioning_class.o: ../threads.h ../threads/threads_kernel.h ../platform.h
conditioning_class.o: ../threads/posix.h ../threads/threads_kernel_2.h
conditioning_class.o: ../threads/threads_kernel_abstract.h
conditioning_class.o: /usr/include/pthread.h /usr/include/features.h
conditioning_class.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
conditioning_class.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
conditioning_class.o: /usr/include/sched.h /usr/include/bits/types.h
conditioning_class.o: /usr/include/bits/typesizes.h /usr/include/time.h
conditioning_class.o: /usr/include/bits/sched.h /usr/include/signal.h
conditioning_class.o: /usr/include/bits/sigset.h
conditioning_class.o: /usr/include/bits/pthreadtypes.h
conditioning_class.o: /usr/include/bits/setjmp.h /usr/include/errno.h
conditioning_class.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
conditioning_class.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
conditioning_class.o: /usr/include/asm-generic/errno.h
conditioning_class.o: /usr/include/asm-generic/errno-base.h
conditioning_class.o: /usr/include/sys/time.h /usr/include/bits/time.h
conditioning_class.o: /usr/include/sys/select.h /usr/include/bits/select.h
conditioning_class.o: ../threads/threads_kernel_shared.h
conditioning_class.o: ../threads/auto_mutex_extension.h
conditioning_class.o: ../threads/threads_kernel.h
conditioning_class.o: ../threads/rmutex_extension.h
conditioning_class.o: ../threads/rmutex_extension_abstract.h
conditioning_class.o: ../threads/auto_mutex_extension_abstract.h
conditioning_class.o: ../binary_search_tree.h
conditioning_class.o: ../binary_search_tree/binary_search_tree_kernel_1.h
conditioning_class.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
conditioning_class.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
conditioning_class.o: ../interfaces/remover.h ../serialize.h ../algs.h
conditioning_class.o: ../uintn.h ../interfaces/enumerable.h
conditioning_class.o: ../interfaces/map_pair.h ../enable_if.h
conditioning_class.o: ../binary_search_tree/binary_search_tree_kernel_2.h
conditioning_class.o: ../binary_search_tree/binary_search_tree_kernel_c.h
conditioning_class.o: ../member_function_pointer.h
conditioning_class.o: ../member_function_pointer/member_function_pointer_kernel_1.h
conditioning_class.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
conditioning_class.o: ../member_function_pointer/member_function_pointer_kernel_c.h
conditioning_class.o: ../memory_manager.h
conditioning_class.o: ../memory_manager/memory_manager_kernel_1.h
conditioning_class.o: ../memory_manager/memory_manager_kernel_abstract.h
conditioning_class.o: ../memory_manager/memory_manager_kernel_2.h
conditioning_class.o: ../memory_manager/memory_manager_kernel_3.h
conditioning_class.o: ../memory_manager/memory_manager_kernel_2.h
conditioning_class.o: ../binary_search_tree/binary_search_tree_kernel_2.h
conditioning_class.o: ../queue.h ../queue/queue_kernel_1.h
conditioning_class.o: ../queue/queue_kernel_abstract.h
conditioning_class.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
conditioning_class.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h
conditioning_class.o: ../sort.h ../set.h ../set/set_kernel_1.h
conditioning_class.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
conditioning_class.o: binary_search_tree.h ../../dlib/memory_manager_global.h
conditioning_class.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
conditioning_class.o: ../memory_manager/memory_manager_kernel_abstract.h
conditioning_class.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
conditioning_class.o: ../../dlib/memory_manager_stateless.h
conditioning_class.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
conditioning_class.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
conditioning_class.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
conditioning_class.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
conditioning_class.o: ../set/set_compare_abstract.h
conditioning_class.o: ../threads/auto_mutex_extension.h
conditioning_class.o: ../threads/auto_unlock_extension.h
conditioning_class.o: ../threads/auto_unlock_extension_abstract.h
conditioning_class.o: ../threads/create_new_thread_extension.h
conditioning_class.o: ../threads/create_new_thread_extension_abstract.h
conditioning_class.o: ../threads/multithreaded_object_extension.h
conditioning_class.o: ../threads/multithreaded_object_extension_abstract.h
conditioning_class.o: ../threads/rsignaler_extension.h
conditioning_class.o: ../threads/rsignaler_extension_abstract.h ../map.h
conditioning_class.o: ../threads/rmutex_extension.h
conditioning_class.o: ../threads/rsignaler_extension.h
conditioning_class.o: ../threads/threaded_object_extension.h
conditioning_class.o: ../threads/threaded_object_extension_abstract.h
conditioning_class.o: ../threads/thread_specific_data_extension.h
conditioning_class.o: ../threads/thread_specific_data_extension_abstract.h
conditioning_class.o: ../threads/thread_function_extension.h
conditioning_class.o: ../threads/thread_function_extension_abstract.h
conditioning_class.o: ../threads/threaded_object_extension.h ../misc_api.h
conditioning_class.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
conditioning_class.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
conditioning_class.o: ../../dlib/logger/logger_kernel_abstract.h
conditioning_class.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
conditioning_class.o: ../noncopyable.h
conditioning_class.o: ../smart_pointers/scoped_ptr_abstract.h
conditioning_class.o: ../smart_pointers/shared_ptr.h
conditioning_class.o: ../smart_pointers/shared_ptr_abstract.h
conditioning_class.o: ../smart_pointers/weak_ptr.h
conditioning_class.o: ../smart_pointers/shared_ptr.h
conditioning_class.o: ../smart_pointers/weak_ptr_abstract.h
conditioning_class.o: ../../dlib/logger/extra_logger_headers.h
conditioning_class.o: ../../dlib/logger/logger_kernel_1.h
conditioning_class.o: ../../dlib/logger/logger_config_file.h
conditioning_class.o: ../config_reader.h
conditioning_class.o: ../config_reader/config_reader_kernel_1.h
conditioning_class.o: ../config_reader/config_reader_kernel_abstract.h
conditioning_class.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
conditioning_class.o: ../../dlib/map/map_kernel_abstract.h
conditioning_class.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
conditioning_class.o: ../tokenizer/tokenizer_kernel_1.h
conditioning_class.o: ../tokenizer/tokenizer_kernel_abstract.h
conditioning_class.o: ../tokenizer/tokenizer_kernel_c.h
conditioning_class.o: ../config_reader/config_reader_thread_safe_1.h
conditioning_class.o: ../config_reader/config_reader_thread_safe_abstract.h
conditioning_class.o: ../../dlib/assert.h ../../dlib/algs.h
conditioning_class.o: conditioning_class.h
conditioning_class_c.o: ../../dlib/conditioning_class.h
conditioning_class_c.o: ../../dlib/conditioning_class/conditioning_class_kernel_1.h
conditioning_class_c.o: ../../dlib/conditioning_class/conditioning_class_kernel_abstract.h
conditioning_class_c.o: ../assert.h ../algs.h ../platform.h ../assert.h
conditioning_class_c.o: ../error.h ../noncopyable.h
conditioning_class_c.o: ../../dlib/conditioning_class/conditioning_class_kernel_2.h
conditioning_class_c.o: ../../dlib/conditioning_class/conditioning_class_kernel_3.h
conditioning_class_c.o: ../../dlib/conditioning_class/conditioning_class_kernel_4.h
conditioning_class_c.o: ../../dlib/conditioning_class/conditioning_class_kernel_c.h
conditioning_class_c.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
conditioning_class_c.o: ../../dlib/logger.h
conditioning_class_c.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
conditioning_class_c.o: ../threads/threads_kernel.h ../platform.h
conditioning_class_c.o: ../threads/posix.h ../threads/threads_kernel_2.h
conditioning_class_c.o: ../threads/threads_kernel_abstract.h
conditioning_class_c.o: /usr/include/pthread.h /usr/include/features.h
conditioning_class_c.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
conditioning_class_c.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
conditioning_class_c.o: /usr/include/sched.h /usr/include/bits/types.h
conditioning_class_c.o: /usr/include/bits/typesizes.h /usr/include/time.h
conditioning_class_c.o: /usr/include/bits/sched.h /usr/include/signal.h
conditioning_class_c.o: /usr/include/bits/sigset.h
conditioning_class_c.o: /usr/include/bits/pthreadtypes.h
conditioning_class_c.o: /usr/include/bits/setjmp.h /usr/include/errno.h
conditioning_class_c.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
conditioning_class_c.o: /usr/include/asm/errno.h
conditioning_class_c.o: /usr/include/asm-i386/errno.h
conditioning_class_c.o: /usr/include/asm-generic/errno.h
conditioning_class_c.o: /usr/include/asm-generic/errno-base.h
conditioning_class_c.o: /usr/include/sys/time.h /usr/include/bits/time.h
conditioning_class_c.o: /usr/include/sys/select.h /usr/include/bits/select.h
conditioning_class_c.o: ../threads/threads_kernel_shared.h
conditioning_class_c.o: ../threads/auto_mutex_extension.h
conditioning_class_c.o: ../threads/threads_kernel.h
conditioning_class_c.o: ../threads/rmutex_extension.h
conditioning_class_c.o: ../threads/rmutex_extension_abstract.h
conditioning_class_c.o: ../threads/auto_mutex_extension_abstract.h
conditioning_class_c.o: ../binary_search_tree.h
conditioning_class_c.o: ../binary_search_tree/binary_search_tree_kernel_1.h
conditioning_class_c.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
conditioning_class_c.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
conditioning_class_c.o: ../interfaces/remover.h ../serialize.h ../algs.h
conditioning_class_c.o: ../uintn.h ../interfaces/enumerable.h
conditioning_class_c.o: ../interfaces/map_pair.h ../enable_if.h
conditioning_class_c.o: ../binary_search_tree/binary_search_tree_kernel_2.h
conditioning_class_c.o: ../binary_search_tree/binary_search_tree_kernel_c.h
conditioning_class_c.o: ../member_function_pointer.h
conditioning_class_c.o: ../member_function_pointer/member_function_pointer_kernel_1.h
conditioning_class_c.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
conditioning_class_c.o: ../member_function_pointer/member_function_pointer_kernel_c.h
conditioning_class_c.o: ../memory_manager.h
conditioning_class_c.o: ../memory_manager/memory_manager_kernel_1.h
conditioning_class_c.o: ../memory_manager/memory_manager_kernel_abstract.h
conditioning_class_c.o: ../memory_manager/memory_manager_kernel_2.h
conditioning_class_c.o: ../memory_manager/memory_manager_kernel_3.h
conditioning_class_c.o: ../memory_manager/memory_manager_kernel_2.h
conditioning_class_c.o: ../binary_search_tree/binary_search_tree_kernel_2.h
conditioning_class_c.o: ../queue.h ../queue/queue_kernel_1.h
conditioning_class_c.o: ../queue/queue_kernel_abstract.h
conditioning_class_c.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
conditioning_class_c.o: ../queue/queue_sort_1.h
conditioning_class_c.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
conditioning_class_c.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
conditioning_class_c.o: ../set/set_kernel_c.h binary_search_tree.h
conditioning_class_c.o: ../../dlib/memory_manager_global.h
conditioning_class_c.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
conditioning_class_c.o: ../memory_manager/memory_manager_kernel_abstract.h
conditioning_class_c.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
conditioning_class_c.o: ../../dlib/memory_manager_stateless.h
conditioning_class_c.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
conditioning_class_c.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
conditioning_class_c.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
conditioning_class_c.o: ../../dlib/binary_search_tree.h
conditioning_class_c.o: ../set/set_compare_1.h ../set/set_compare_abstract.h
conditioning_class_c.o: ../threads/auto_mutex_extension.h
conditioning_class_c.o: ../threads/auto_unlock_extension.h
conditioning_class_c.o: ../threads/auto_unlock_extension_abstract.h
conditioning_class_c.o: ../threads/create_new_thread_extension.h
conditioning_class_c.o: ../threads/create_new_thread_extension_abstract.h
conditioning_class_c.o: ../threads/multithreaded_object_extension.h
conditioning_class_c.o: ../threads/multithreaded_object_extension_abstract.h
conditioning_class_c.o: ../threads/rsignaler_extension.h
conditioning_class_c.o: ../threads/rsignaler_extension_abstract.h ../map.h
conditioning_class_c.o: ../threads/rmutex_extension.h
conditioning_class_c.o: ../threads/rsignaler_extension.h
conditioning_class_c.o: ../threads/threaded_object_extension.h
conditioning_class_c.o: ../threads/threaded_object_extension_abstract.h
conditioning_class_c.o: ../threads/thread_specific_data_extension.h
conditioning_class_c.o: ../threads/thread_specific_data_extension_abstract.h
conditioning_class_c.o: ../threads/thread_function_extension.h
conditioning_class_c.o: ../threads/thread_function_extension_abstract.h
conditioning_class_c.o: ../threads/threaded_object_extension.h ../misc_api.h
conditioning_class_c.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
conditioning_class_c.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
conditioning_class_c.o: ../../dlib/logger/logger_kernel_abstract.h
conditioning_class_c.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
conditioning_class_c.o: ../noncopyable.h
conditioning_class_c.o: ../smart_pointers/scoped_ptr_abstract.h
conditioning_class_c.o: ../smart_pointers/shared_ptr.h
conditioning_class_c.o: ../smart_pointers/shared_ptr_abstract.h
conditioning_class_c.o: ../smart_pointers/weak_ptr.h
conditioning_class_c.o: ../smart_pointers/shared_ptr.h
conditioning_class_c.o: ../smart_pointers/weak_ptr_abstract.h
conditioning_class_c.o: ../../dlib/logger/extra_logger_headers.h
conditioning_class_c.o: ../../dlib/logger/logger_kernel_1.h
conditioning_class_c.o: ../../dlib/logger/logger_config_file.h
conditioning_class_c.o: ../config_reader.h
conditioning_class_c.o: ../config_reader/config_reader_kernel_1.h
conditioning_class_c.o: ../config_reader/config_reader_kernel_abstract.h
conditioning_class_c.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
conditioning_class_c.o: ../../dlib/map/map_kernel_abstract.h
conditioning_class_c.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
conditioning_class_c.o: ../tokenizer/tokenizer_kernel_1.h
conditioning_class_c.o: ../tokenizer/tokenizer_kernel_abstract.h
conditioning_class_c.o: ../tokenizer/tokenizer_kernel_c.h
conditioning_class_c.o: ../config_reader/config_reader_thread_safe_1.h
conditioning_class_c.o: ../config_reader/config_reader_thread_safe_abstract.h
conditioning_class_c.o: ../../dlib/assert.h ../../dlib/algs.h
conditioning_class_c.o: conditioning_class.h
config_reader.o: ../../dlib/config_reader.h
config_reader.o: ../config_reader/config_reader_kernel_1.h
config_reader.o: ../config_reader/config_reader_kernel_abstract.h ../algs.h
config_reader.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
config_reader.o: ../interfaces/enumerable.h ../../dlib/map.h
config_reader.o: ../../dlib/map/map_kernel_1.h
config_reader.o: ../../dlib/map/map_kernel_abstract.h
config_reader.o: ../interfaces/map_pair.h ../interfaces/remover.h
config_reader.o: ../serialize.h ../algs.h ../uintn.h
config_reader.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
config_reader.o: ../enable_if.h ../memory_manager.h
config_reader.o: ../memory_manager/memory_manager_kernel_1.h
config_reader.o: ../memory_manager/memory_manager_kernel_abstract.h
config_reader.o: ../assert.h ../memory_manager/memory_manager_kernel_2.h
config_reader.o: ../memory_manager/memory_manager_kernel_3.h
config_reader.o: ../memory_manager/memory_manager_kernel_2.h
config_reader.o: ../binary_search_tree/binary_search_tree_kernel_2.h
config_reader.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
config_reader.o: ../../dlib/map/map_kernel_c.h binary_search_tree.h
config_reader.o: ../../dlib/memory_manager_global.h
config_reader.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
config_reader.o: ../memory_manager/memory_manager_kernel_abstract.h
config_reader.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
config_reader.o: ../../dlib/memory_manager.h
config_reader.o: ../../dlib/memory_manager_stateless.h
config_reader.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
config_reader.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
config_reader.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
config_reader.o: ../threads.h ../threads/threads_kernel.h ../platform.h
config_reader.o: ../threads/posix.h ../threads/threads_kernel_2.h
config_reader.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
config_reader.o: /usr/include/features.h /usr/include/sys/cdefs.h
config_reader.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
config_reader.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
config_reader.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
config_reader.o: /usr/include/time.h /usr/include/bits/sched.h
config_reader.o: /usr/include/signal.h /usr/include/bits/sigset.h
config_reader.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
config_reader.o: /usr/include/errno.h /usr/include/bits/errno.h
config_reader.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
config_reader.o: /usr/include/asm-i386/errno.h
config_reader.o: /usr/include/asm-generic/errno.h
config_reader.o: /usr/include/asm-generic/errno-base.h
config_reader.o: /usr/include/sys/time.h /usr/include/bits/time.h
config_reader.o: /usr/include/sys/select.h /usr/include/bits/select.h
config_reader.o: ../threads/threads_kernel_shared.h
config_reader.o: ../threads/auto_mutex_extension.h
config_reader.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
config_reader.o: ../threads/rmutex_extension_abstract.h
config_reader.o: ../threads/auto_mutex_extension_abstract.h
config_reader.o: ../binary_search_tree.h
config_reader.o: ../binary_search_tree/binary_search_tree_kernel_1.h
config_reader.o: ../binary_search_tree/binary_search_tree_kernel_2.h
config_reader.o: ../binary_search_tree/binary_search_tree_kernel_c.h
config_reader.o: ../member_function_pointer.h
config_reader.o: ../member_function_pointer/member_function_pointer_kernel_1.h
config_reader.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
config_reader.o: ../member_function_pointer/member_function_pointer_kernel_c.h
config_reader.o: ../queue.h ../queue/queue_kernel_1.h
config_reader.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
config_reader.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
config_reader.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
config_reader.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
config_reader.o: ../set/set_kernel_c.h ../set/set_compare_1.h
config_reader.o: ../set/set_compare_abstract.h
config_reader.o: ../threads/auto_mutex_extension.h
config_reader.o: ../threads/auto_unlock_extension.h
config_reader.o: ../threads/auto_unlock_extension_abstract.h
config_reader.o: ../threads/create_new_thread_extension.h
config_reader.o: ../threads/create_new_thread_extension_abstract.h
config_reader.o: ../threads/multithreaded_object_extension.h
config_reader.o: ../threads/multithreaded_object_extension_abstract.h
config_reader.o: ../threads/rsignaler_extension.h
config_reader.o: ../threads/rsignaler_extension_abstract.h ../map.h
config_reader.o: ../threads/rmutex_extension.h
config_reader.o: ../threads/rsignaler_extension.h
config_reader.o: ../threads/threaded_object_extension.h
config_reader.o: ../threads/threaded_object_extension_abstract.h
config_reader.o: ../threads/thread_specific_data_extension.h
config_reader.o: ../threads/thread_specific_data_extension_abstract.h
config_reader.o: ../threads/thread_function_extension.h
config_reader.o: ../threads/thread_function_extension_abstract.h
config_reader.o: ../threads/threaded_object_extension.h
config_reader.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
config_reader.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
config_reader.o: ../misc_api.h ../misc_api/posix.h
config_reader.o: ../misc_api/misc_api_kernel_2.h
config_reader.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
config_reader.o: ../../dlib/logger/logger_kernel_abstract.h
config_reader.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
config_reader.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
config_reader.o: ../smart_pointers/shared_ptr.h
config_reader.o: ../smart_pointers/shared_ptr_abstract.h
config_reader.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
config_reader.o: ../smart_pointers/weak_ptr_abstract.h
config_reader.o: ../../dlib/logger/extra_logger_headers.h
config_reader.o: ../../dlib/logger/logger_kernel_1.h
config_reader.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
config_reader.o: ../tokenizer.h ../tokenizer/tokenizer_kernel_1.h
config_reader.o: ../tokenizer/tokenizer_kernel_abstract.h
config_reader.o: ../tokenizer/tokenizer_kernel_c.h
config_reader.o: ../config_reader/config_reader_thread_safe_1.h
config_reader.o: ../config_reader/config_reader_thread_safe_abstract.h
config_reader.o: ../../dlib/assert.h ../../dlib/algs.h
directed_graph.o: ../../dlib/directed_graph.h
directed_graph.o: ../../dlib/directed_graph/directed_graph_kernel_1.h
directed_graph.o: ../serialize.h ../algs.h ../assert.h ../error.h ../uintn.h
directed_graph.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
directed_graph.o: ../enable_if.h ../noncopyable.h ../std_allocator.h
directed_graph.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
directed_graph.o: ../algs.h ../platform.h ../noncopyable.h
directed_graph.o: ../smart_pointers/scoped_ptr_abstract.h
directed_graph.o: ../smart_pointers/shared_ptr.h
directed_graph.o: ../smart_pointers/shared_ptr_abstract.h
directed_graph.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
directed_graph.o: ../smart_pointers/weak_ptr_abstract.h ../memory_manager.h
directed_graph.o: ../memory_manager/memory_manager_kernel_1.h
directed_graph.o: ../memory_manager/memory_manager_kernel_abstract.h
directed_graph.o: ../assert.h ../memory_manager/memory_manager_kernel_2.h
directed_graph.o: ../memory_manager/memory_manager_kernel_3.h
directed_graph.o: ../memory_manager/memory_manager_kernel_2.h
directed_graph.o: ../binary_search_tree/binary_search_tree_kernel_2.h
directed_graph.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
directed_graph.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
directed_graph.o: ../interfaces/remover.h
directed_graph.o: ../../dlib/directed_graph/directed_graph_kernel_abstract.h
directed_graph.o: ../is_kind.h ../../dlib/memory_manager.h ../../dlib/graph.h
directed_graph.o: ../../dlib/graph/graph_kernel_1.h
directed_graph.o: ../../dlib/graph/graph_kernel_abstract.h
directed_graph.o: ../../dlib/graph_utils.h
directed_graph.o: ../../dlib/graph_utils/graph_utils.h
directed_graph.o: ../../dlib/graph_utils/graph_utils_abstract.h
directed_graph.o: ../enable_if.h ../set.h ../set/set_kernel_1.h
directed_graph.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
directed_graph.o: binary_search_tree.h ../../dlib/memory_manager_global.h
directed_graph.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
directed_graph.o: ../memory_manager/memory_manager_kernel_abstract.h
directed_graph.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
directed_graph.o: ../../dlib/memory_manager_stateless.h
directed_graph.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
directed_graph.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
directed_graph.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
directed_graph.o: ../threads.h ../threads/threads_kernel.h ../platform.h
directed_graph.o: ../threads/posix.h ../threads/threads_kernel_2.h
directed_graph.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
directed_graph.o: /usr/include/features.h /usr/include/sys/cdefs.h
directed_graph.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
directed_graph.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
directed_graph.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
directed_graph.o: /usr/include/time.h /usr/include/bits/sched.h
directed_graph.o: /usr/include/signal.h /usr/include/bits/sigset.h
directed_graph.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
directed_graph.o: /usr/include/errno.h /usr/include/bits/errno.h
directed_graph.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
directed_graph.o: /usr/include/asm-i386/errno.h
directed_graph.o: /usr/include/asm-generic/errno.h
directed_graph.o: /usr/include/asm-generic/errno-base.h
directed_graph.o: /usr/include/sys/time.h /usr/include/bits/time.h
directed_graph.o: /usr/include/sys/select.h /usr/include/bits/select.h
directed_graph.o: ../threads/threads_kernel_shared.h
directed_graph.o: ../threads/auto_mutex_extension.h
directed_graph.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
directed_graph.o: ../threads/rmutex_extension_abstract.h
directed_graph.o: ../threads/auto_mutex_extension_abstract.h
directed_graph.o: ../binary_search_tree.h
directed_graph.o: ../binary_search_tree/binary_search_tree_kernel_1.h
directed_graph.o: ../binary_search_tree/binary_search_tree_kernel_2.h
directed_graph.o: ../binary_search_tree/binary_search_tree_kernel_c.h
directed_graph.o: ../member_function_pointer.h
directed_graph.o: ../member_function_pointer/member_function_pointer_kernel_1.h
directed_graph.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
directed_graph.o: ../member_function_pointer/member_function_pointer_kernel_c.h
directed_graph.o: ../queue.h ../queue/queue_kernel_1.h
directed_graph.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
directed_graph.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
directed_graph.o: ../queue/queue_sort_abstract.h ../sort.h
directed_graph.o: ../threads/auto_mutex_extension.h
directed_graph.o: ../threads/auto_unlock_extension.h
directed_graph.o: ../threads/auto_unlock_extension_abstract.h
directed_graph.o: ../threads/create_new_thread_extension.h
directed_graph.o: ../threads/create_new_thread_extension_abstract.h
directed_graph.o: ../threads/multithreaded_object_extension.h
directed_graph.o: ../threads/multithreaded_object_extension_abstract.h
directed_graph.o: ../threads/rsignaler_extension.h
directed_graph.o: ../threads/rsignaler_extension_abstract.h ../map.h
directed_graph.o: ../threads/rmutex_extension.h
directed_graph.o: ../threads/rsignaler_extension.h
directed_graph.o: ../threads/threaded_object_extension.h
directed_graph.o: ../threads/threaded_object_extension_abstract.h
directed_graph.o: ../threads/thread_specific_data_extension.h
directed_graph.o: ../threads/thread_specific_data_extension_abstract.h
directed_graph.o: ../threads/thread_function_extension.h
directed_graph.o: ../threads/thread_function_extension_abstract.h
directed_graph.o: ../threads/threaded_object_extension.h
directed_graph.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
directed_graph.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
directed_graph.o: ../misc_api.h ../misc_api/posix.h
directed_graph.o: ../misc_api/misc_api_kernel_2.h
directed_graph.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
directed_graph.o: ../../dlib/logger/logger_kernel_abstract.h
directed_graph.o: ../../dlib/logger/extra_logger_headers.h
directed_graph.o: ../../dlib/logger/logger_kernel_1.h
directed_graph.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
directed_graph.o: ../config_reader/config_reader_kernel_1.h
directed_graph.o: ../config_reader/config_reader_kernel_abstract.h
directed_graph.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
directed_graph.o: ../../dlib/map/map_kernel_abstract.h
directed_graph.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
directed_graph.o: ../tokenizer/tokenizer_kernel_1.h
directed_graph.o: ../tokenizer/tokenizer_kernel_abstract.h
directed_graph.o: ../tokenizer/tokenizer_kernel_c.h
directed_graph.o: ../config_reader/config_reader_thread_safe_1.h
directed_graph.o: ../config_reader/config_reader_thread_safe_abstract.h
directed_graph.o: ../../dlib/assert.h ../../dlib/algs.h
directed_graph.o: ../set/set_compare_1.h ../set/set_compare_abstract.h
directed_graph.o: ../set_utils.h ../set_utils/set_utils.h
directed_graph.o: ../set_utils/set_utils_abstract.h ../../dlib/set.h
graph.o: ../../dlib/graph.h ../../dlib/graph/graph_kernel_1.h ../serialize.h
graph.o: ../algs.h ../assert.h ../error.h ../uintn.h
graph.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
graph.o: ../noncopyable.h ../std_allocator.h ../smart_pointers.h
graph.o: ../smart_pointers/scoped_ptr.h ../algs.h ../platform.h
graph.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
graph.o: ../smart_pointers/shared_ptr.h
graph.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
graph.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
graph.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
graph.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
graph.o: ../memory_manager/memory_manager_kernel_2.h
graph.o: ../memory_manager/memory_manager_kernel_3.h
graph.o: ../memory_manager/memory_manager_kernel_2.h
graph.o: ../binary_search_tree/binary_search_tree_kernel_2.h
graph.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
graph.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
graph.o: ../interfaces/remover.h ../../dlib/graph/graph_kernel_abstract.h
graph.o: ../is_kind.h ../../dlib/memory_manager.h ../../dlib/graph_utils.h
graph.o: ../../dlib/graph_utils/graph_utils.h
graph.o: ../../dlib/graph_utils/graph_utils_abstract.h ../enable_if.h
graph.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
graph.o: ../set/set_kernel_c.h binary_search_tree.h
graph.o: ../../dlib/memory_manager_global.h
graph.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
graph.o: ../memory_manager/memory_manager_kernel_abstract.h
graph.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
graph.o: ../../dlib/memory_manager_stateless.h
graph.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
graph.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
graph.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
graph.o: ../threads.h ../threads/threads_kernel.h ../platform.h
graph.o: ../threads/posix.h ../threads/threads_kernel_2.h
graph.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
graph.o: /usr/include/features.h /usr/include/sys/cdefs.h
graph.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
graph.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
graph.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
graph.o: /usr/include/time.h /usr/include/bits/sched.h /usr/include/signal.h
graph.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
graph.o: /usr/include/bits/setjmp.h /usr/include/errno.h
graph.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
graph.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
graph.o: /usr/include/asm-generic/errno.h
graph.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
graph.o: /usr/include/bits/time.h /usr/include/sys/select.h
graph.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
graph.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
graph.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
graph.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
graph.o: ../binary_search_tree/binary_search_tree_kernel_1.h
graph.o: ../binary_search_tree/binary_search_tree_kernel_2.h
graph.o: ../binary_search_tree/binary_search_tree_kernel_c.h
graph.o: ../member_function_pointer.h
graph.o: ../member_function_pointer/member_function_pointer_kernel_1.h
graph.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
graph.o: ../member_function_pointer/member_function_pointer_kernel_c.h
graph.o: ../queue.h ../queue/queue_kernel_1.h
graph.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
graph.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
graph.o: ../queue/queue_sort_abstract.h ../sort.h
graph.o: ../threads/auto_mutex_extension.h ../threads/auto_unlock_extension.h
graph.o: ../threads/auto_unlock_extension_abstract.h
graph.o: ../threads/create_new_thread_extension.h
graph.o: ../threads/create_new_thread_extension_abstract.h
graph.o: ../threads/multithreaded_object_extension.h
graph.o: ../threads/multithreaded_object_extension_abstract.h
graph.o: ../threads/rsignaler_extension.h
graph.o: ../threads/rsignaler_extension_abstract.h ../map.h
graph.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
graph.o: ../threads/threaded_object_extension.h
graph.o: ../threads/threaded_object_extension_abstract.h
graph.o: ../threads/thread_specific_data_extension.h
graph.o: ../threads/thread_specific_data_extension_abstract.h
graph.o: ../threads/thread_function_extension.h
graph.o: ../threads/thread_function_extension_abstract.h
graph.o: ../threads/threaded_object_extension.h
graph.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
graph.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
graph.o: ../misc_api.h ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
graph.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
graph.o: ../../dlib/logger/logger_kernel_abstract.h
graph.o: ../../dlib/logger/extra_logger_headers.h
graph.o: ../../dlib/logger/logger_kernel_1.h
graph.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
graph.o: ../config_reader/config_reader_kernel_1.h
graph.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
graph.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
graph.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
graph.o: ../tokenizer/tokenizer_kernel_1.h
graph.o: ../tokenizer/tokenizer_kernel_abstract.h
graph.o: ../tokenizer/tokenizer_kernel_c.h
graph.o: ../config_reader/config_reader_thread_safe_1.h
graph.o: ../config_reader/config_reader_thread_safe_abstract.h
graph.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
graph.o: ../set/set_compare_abstract.h ../set_utils.h
graph.o: ../set_utils/set_utils.h ../set_utils/set_utils_abstract.h
graph.o: ../../dlib/set.h
entropy_coder.o: ../../dlib/entropy_encoder.h
entropy_coder.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_1.h
entropy_coder.o: ../algs.h ../platform.h ../assert.h ../error.h
entropy_coder.o: ../noncopyable.h
entropy_coder.o: ../entropy_encoder/entropy_encoder_kernel_abstract.h
entropy_coder.o: ../uintn.h
entropy_coder.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_2.h
entropy_coder.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_c.h
entropy_coder.o: ../assert.h ../../dlib/entropy_decoder.h
entropy_coder.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_1.h
entropy_coder.o: ../entropy_decoder/entropy_decoder_kernel_abstract.h
entropy_coder.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_2.h
entropy_coder.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_c.h
entropy_coder.o: tester.h ../../dlib/map.h ../../dlib/logger.h
entropy_coder.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
entropy_coder.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
entropy_coder.o: ../threads/threads_kernel_2.h
entropy_coder.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
entropy_coder.o: /usr/include/features.h /usr/include/sys/cdefs.h
entropy_coder.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
entropy_coder.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
entropy_coder.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
entropy_coder.o: /usr/include/time.h /usr/include/bits/sched.h
entropy_coder.o: /usr/include/signal.h /usr/include/bits/sigset.h
entropy_coder.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
entropy_coder.o: /usr/include/errno.h /usr/include/bits/errno.h
entropy_coder.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
entropy_coder.o: /usr/include/asm-i386/errno.h
entropy_coder.o: /usr/include/asm-generic/errno.h
entropy_coder.o: /usr/include/asm-generic/errno-base.h
entropy_coder.o: /usr/include/sys/time.h /usr/include/bits/time.h
entropy_coder.o: /usr/include/sys/select.h /usr/include/bits/select.h
entropy_coder.o: ../threads/threads_kernel_shared.h
entropy_coder.o: ../threads/auto_mutex_extension.h
entropy_coder.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
entropy_coder.o: ../threads/rmutex_extension_abstract.h
entropy_coder.o: ../threads/auto_mutex_extension_abstract.h
entropy_coder.o: ../binary_search_tree.h
entropy_coder.o: ../binary_search_tree/binary_search_tree_kernel_1.h
entropy_coder.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
entropy_coder.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
entropy_coder.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
entropy_coder.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
entropy_coder.o: ../enable_if.h
entropy_coder.o: ../binary_search_tree/binary_search_tree_kernel_2.h
entropy_coder.o: ../binary_search_tree/binary_search_tree_kernel_c.h
entropy_coder.o: ../../dlib/memory_manager.h ../member_function_pointer.h
entropy_coder.o: ../member_function_pointer/member_function_pointer_kernel_1.h
entropy_coder.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
entropy_coder.o: ../member_function_pointer/member_function_pointer_kernel_c.h
entropy_coder.o: ../memory_manager.h
entropy_coder.o: ../memory_manager/memory_manager_kernel_1.h
entropy_coder.o: ../memory_manager/memory_manager_kernel_abstract.h
entropy_coder.o: ../memory_manager/memory_manager_kernel_2.h
entropy_coder.o: ../memory_manager/memory_manager_kernel_3.h
entropy_coder.o: ../memory_manager/memory_manager_kernel_2.h
entropy_coder.o: ../binary_search_tree/binary_search_tree_kernel_2.h
entropy_coder.o: ../queue.h ../queue/queue_kernel_1.h
entropy_coder.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
entropy_coder.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
entropy_coder.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
entropy_coder.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
entropy_coder.o: ../set/set_kernel_c.h binary_search_tree.h
entropy_coder.o: ../../dlib/memory_manager_global.h
entropy_coder.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
entropy_coder.o: ../memory_manager/memory_manager_kernel_abstract.h
entropy_coder.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
entropy_coder.o: ../../dlib/memory_manager_stateless.h
entropy_coder.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
entropy_coder.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
entropy_coder.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
entropy_coder.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
entropy_coder.o: ../set/set_compare_abstract.h
entropy_coder.o: ../threads/auto_mutex_extension.h
entropy_coder.o: ../threads/auto_unlock_extension.h
entropy_coder.o: ../threads/auto_unlock_extension_abstract.h
entropy_coder.o: ../threads/create_new_thread_extension.h
entropy_coder.o: ../threads/create_new_thread_extension_abstract.h
entropy_coder.o: ../threads/multithreaded_object_extension.h
entropy_coder.o: ../threads/multithreaded_object_extension_abstract.h
entropy_coder.o: ../threads/rsignaler_extension.h
entropy_coder.o: ../threads/rsignaler_extension_abstract.h ../map.h
entropy_coder.o: ../threads/rmutex_extension.h
entropy_coder.o: ../threads/rsignaler_extension.h
entropy_coder.o: ../threads/threaded_object_extension.h
entropy_coder.o: ../threads/threaded_object_extension_abstract.h
entropy_coder.o: ../threads/thread_specific_data_extension.h
entropy_coder.o: ../threads/thread_specific_data_extension_abstract.h
entropy_coder.o: ../threads/thread_function_extension.h
entropy_coder.o: ../threads/thread_function_extension_abstract.h
entropy_coder.o: ../threads/threaded_object_extension.h ../misc_api.h
entropy_coder.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
entropy_coder.o: ../misc_api/misc_api_kernel_abstract.h
entropy_coder.o: ../../dlib/logger/logger_kernel_abstract.h
entropy_coder.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
entropy_coder.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
entropy_coder.o: ../smart_pointers/shared_ptr.h
entropy_coder.o: ../smart_pointers/shared_ptr_abstract.h
entropy_coder.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
entropy_coder.o: ../smart_pointers/weak_ptr_abstract.h
entropy_coder.o: ../../dlib/logger/extra_logger_headers.h
entropy_coder.o: ../../dlib/logger/logger_kernel_1.h
entropy_coder.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
entropy_coder.o: ../config_reader/config_reader_kernel_1.h
entropy_coder.o: ../config_reader/config_reader_kernel_abstract.h
entropy_coder.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
entropy_coder.o: ../../dlib/map/map_kernel_abstract.h
entropy_coder.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
entropy_coder.o: ../tokenizer/tokenizer_kernel_1.h
entropy_coder.o: ../tokenizer/tokenizer_kernel_abstract.h
entropy_coder.o: ../tokenizer/tokenizer_kernel_c.h
entropy_coder.o: ../config_reader/config_reader_thread_safe_1.h
entropy_coder.o: ../config_reader/config_reader_thread_safe_abstract.h
entropy_coder.o: ../../dlib/assert.h ../../dlib/algs.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_1.h
entropy_encoder_model.o: ../algs.h ../platform.h ../assert.h ../error.h
entropy_encoder_model.o: ../noncopyable.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_abstract.h
entropy_encoder_model.o: ../assert.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_2.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_3.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_4.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_5.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_6.h
entropy_encoder_model.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_c.h
entropy_encoder_model.o: conditioning_class.h ../../dlib/conditioning_class.h
entropy_encoder_model.o: ../../dlib/conditioning_class/conditioning_class_kernel_1.h
entropy_encoder_model.o: ../../dlib/conditioning_class/conditioning_class_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/conditioning_class/conditioning_class_kernel_2.h
entropy_encoder_model.o: ../../dlib/conditioning_class/conditioning_class_kernel_3.h
entropy_encoder_model.o: ../../dlib/conditioning_class/conditioning_class_kernel_4.h
entropy_encoder_model.o: ../../dlib/conditioning_class/conditioning_class_kernel_c.h
entropy_encoder_model.o: ../../dlib/memory_manager.h tester.h
entropy_encoder_model.o: ../../dlib/map.h ../../dlib/logger.h
entropy_encoder_model.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
entropy_encoder_model.o: ../threads/threads_kernel.h ../platform.h
entropy_encoder_model.o: ../threads/posix.h ../threads/threads_kernel_2.h
entropy_encoder_model.o: ../threads/threads_kernel_abstract.h
entropy_encoder_model.o: /usr/include/pthread.h /usr/include/features.h
entropy_encoder_model.o: /usr/include/sys/cdefs.h
entropy_encoder_model.o: /usr/include/bits/wordsize.h
entropy_encoder_model.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
entropy_encoder_model.o: /usr/include/sched.h /usr/include/bits/types.h
entropy_encoder_model.o: /usr/include/bits/typesizes.h /usr/include/time.h
entropy_encoder_model.o: /usr/include/bits/sched.h /usr/include/signal.h
entropy_encoder_model.o: /usr/include/bits/sigset.h
entropy_encoder_model.o: /usr/include/bits/pthreadtypes.h
entropy_encoder_model.o: /usr/include/bits/setjmp.h /usr/include/errno.h
entropy_encoder_model.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
entropy_encoder_model.o: /usr/include/asm/errno.h
entropy_encoder_model.o: /usr/include/asm-i386/errno.h
entropy_encoder_model.o: /usr/include/asm-generic/errno.h
entropy_encoder_model.o: /usr/include/asm-generic/errno-base.h
entropy_encoder_model.o: /usr/include/sys/time.h /usr/include/bits/time.h
entropy_encoder_model.o: /usr/include/sys/select.h /usr/include/bits/select.h
entropy_encoder_model.o: ../threads/threads_kernel_shared.h
entropy_encoder_model.o: ../threads/auto_mutex_extension.h
entropy_encoder_model.o: ../threads/threads_kernel.h
entropy_encoder_model.o: ../threads/rmutex_extension.h
entropy_encoder_model.o: ../threads/rmutex_extension_abstract.h
entropy_encoder_model.o: ../threads/auto_mutex_extension_abstract.h
entropy_encoder_model.o: ../binary_search_tree.h
entropy_encoder_model.o: ../binary_search_tree/binary_search_tree_kernel_1.h
entropy_encoder_model.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
entropy_encoder_model.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
entropy_encoder_model.o: ../interfaces/remover.h ../serialize.h ../algs.h
entropy_encoder_model.o: ../uintn.h ../interfaces/enumerable.h
entropy_encoder_model.o: ../interfaces/map_pair.h ../enable_if.h
entropy_encoder_model.o: ../binary_search_tree/binary_search_tree_kernel_2.h
entropy_encoder_model.o: ../binary_search_tree/binary_search_tree_kernel_c.h
entropy_encoder_model.o: ../member_function_pointer.h
entropy_encoder_model.o: ../member_function_pointer/member_function_pointer_kernel_1.h
entropy_encoder_model.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
entropy_encoder_model.o: ../member_function_pointer/member_function_pointer_kernel_c.h
entropy_encoder_model.o: ../memory_manager.h
entropy_encoder_model.o: ../memory_manager/memory_manager_kernel_1.h
entropy_encoder_model.o: ../memory_manager/memory_manager_kernel_abstract.h
entropy_encoder_model.o: ../memory_manager/memory_manager_kernel_2.h
entropy_encoder_model.o: ../memory_manager/memory_manager_kernel_3.h
entropy_encoder_model.o: ../memory_manager/memory_manager_kernel_2.h
entropy_encoder_model.o: ../binary_search_tree/binary_search_tree_kernel_2.h
entropy_encoder_model.o: ../queue.h ../queue/queue_kernel_1.h
entropy_encoder_model.o: ../queue/queue_kernel_abstract.h
entropy_encoder_model.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
entropy_encoder_model.o: ../queue/queue_sort_1.h
entropy_encoder_model.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
entropy_encoder_model.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
entropy_encoder_model.o: ../set/set_kernel_c.h binary_search_tree.h
entropy_encoder_model.o: ../../dlib/memory_manager_global.h
entropy_encoder_model.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
entropy_encoder_model.o: ../memory_manager/memory_manager_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/memory_manager_stateless.h
entropy_encoder_model.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
entropy_encoder_model.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
entropy_encoder_model.o: ../../dlib/binary_search_tree.h
entropy_encoder_model.o: ../set/set_compare_1.h ../set/set_compare_abstract.h
entropy_encoder_model.o: ../threads/auto_mutex_extension.h
entropy_encoder_model.o: ../threads/auto_unlock_extension.h
entropy_encoder_model.o: ../threads/auto_unlock_extension_abstract.h
entropy_encoder_model.o: ../threads/create_new_thread_extension.h
entropy_encoder_model.o: ../threads/create_new_thread_extension_abstract.h
entropy_encoder_model.o: ../threads/multithreaded_object_extension.h
entropy_encoder_model.o: ../threads/multithreaded_object_extension_abstract.h
entropy_encoder_model.o: ../threads/rsignaler_extension.h
entropy_encoder_model.o: ../threads/rsignaler_extension_abstract.h ../map.h
entropy_encoder_model.o: ../threads/rmutex_extension.h
entropy_encoder_model.o: ../threads/rsignaler_extension.h
entropy_encoder_model.o: ../threads/threaded_object_extension.h
entropy_encoder_model.o: ../threads/threaded_object_extension_abstract.h
entropy_encoder_model.o: ../threads/thread_specific_data_extension.h
entropy_encoder_model.o: ../threads/thread_specific_data_extension_abstract.h
entropy_encoder_model.o: ../threads/thread_function_extension.h
entropy_encoder_model.o: ../threads/thread_function_extension_abstract.h
entropy_encoder_model.o: ../threads/threaded_object_extension.h ../misc_api.h
entropy_encoder_model.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
entropy_encoder_model.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
entropy_encoder_model.o: ../../dlib/logger/logger_kernel_abstract.h
entropy_encoder_model.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
entropy_encoder_model.o: ../noncopyable.h
entropy_encoder_model.o: ../smart_pointers/scoped_ptr_abstract.h
entropy_encoder_model.o: ../smart_pointers/shared_ptr.h
entropy_encoder_model.o: ../smart_pointers/shared_ptr_abstract.h
entropy_encoder_model.o: ../smart_pointers/weak_ptr.h
entropy_encoder_model.o: ../smart_pointers/shared_ptr.h
entropy_encoder_model.o: ../smart_pointers/weak_ptr_abstract.h
entropy_encoder_model.o: ../../dlib/logger/extra_logger_headers.h
entropy_encoder_model.o: ../../dlib/logger/logger_kernel_1.h
entropy_encoder_model.o: ../../dlib/logger/logger_config_file.h
entropy_encoder_model.o: ../config_reader.h
entropy_encoder_model.o: ../config_reader/config_reader_kernel_1.h
entropy_encoder_model.o: ../config_reader/config_reader_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
entropy_encoder_model.o: ../../dlib/map/map_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
entropy_encoder_model.o: ../tokenizer/tokenizer_kernel_1.h
entropy_encoder_model.o: ../tokenizer/tokenizer_kernel_abstract.h
entropy_encoder_model.o: ../tokenizer/tokenizer_kernel_c.h
entropy_encoder_model.o: ../config_reader/config_reader_thread_safe_1.h
entropy_encoder_model.o: ../config_reader/config_reader_thread_safe_abstract.h
entropy_encoder_model.o: ../../dlib/assert.h ../../dlib/algs.h
entropy_encoder_model.o: ../../dlib/sliding_buffer.h
entropy_encoder_model.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_1.h
entropy_encoder_model.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_c.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_1.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_2.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_3.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_4.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_5.h
entropy_encoder_model.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_6.h
entropy_encoder_model.o: ../../dlib/entropy_encoder.h
entropy_encoder_model.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_1.h
entropy_encoder_model.o: ../entropy_encoder/entropy_encoder_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_2.h
entropy_encoder_model.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_c.h
entropy_encoder_model.o: ../../dlib/entropy_decoder.h
entropy_encoder_model.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_1.h
entropy_encoder_model.o: ../entropy_decoder/entropy_decoder_kernel_abstract.h
entropy_encoder_model.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_2.h
entropy_encoder_model.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_c.h
hash_map.o: ../../dlib/hash_map.h ../../dlib/hash_map/hash_map_kernel_1.h
hash_map.o: ../../dlib/hash_map/hash_map_kernel_abstract.h ../algs.h
hash_map.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
hash_map.o: ../general_hash/general_hash.h ../interfaces/enumerable.h
hash_map.o: ../interfaces/map_pair.h ../interfaces/remover.h ../assert.h
hash_map.o: ../serialize.h ../algs.h ../uintn.h ../interfaces/enumerable.h
hash_map.o: ../interfaces/map_pair.h ../enable_if.h ../memory_manager.h
hash_map.o: ../memory_manager/memory_manager_kernel_1.h
hash_map.o: ../memory_manager/memory_manager_kernel_abstract.h
hash_map.o: ../memory_manager/memory_manager_kernel_2.h
hash_map.o: ../memory_manager/memory_manager_kernel_3.h
hash_map.o: ../memory_manager/memory_manager_kernel_2.h
hash_map.o: ../binary_search_tree/binary_search_tree_kernel_2.h
hash_map.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
hash_map.o: ../../dlib/hash_map/hash_map_kernel_c.h ../../dlib/hash_table.h
hash_map.o: ../../dlib/hash_table/hash_table_kernel_1.h
hash_map.o: ../../dlib/hash_table/hash_table_kernel_abstract.h
hash_map.o: ../../dlib/hash_table/hash_table_kernel_2.h
hash_map.o: ../../dlib/hash_table/hash_table_kernel_c.h
hash_map.o: ../../dlib/memory_manager.h binary_search_tree.h
hash_map.o: ../../dlib/memory_manager_global.h
hash_map.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
hash_map.o: ../memory_manager/memory_manager_kernel_abstract.h
hash_map.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
hash_map.o: ../../dlib/memory_manager_stateless.h
hash_map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
hash_map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
hash_map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
hash_map.o: ../threads.h ../threads/threads_kernel.h ../platform.h
hash_map.o: ../threads/posix.h ../threads/threads_kernel_2.h
hash_map.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
hash_map.o: /usr/include/features.h /usr/include/sys/cdefs.h
hash_map.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
hash_map.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
hash_map.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
hash_map.o: /usr/include/time.h /usr/include/bits/sched.h
hash_map.o: /usr/include/signal.h /usr/include/bits/sigset.h
hash_map.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
hash_map.o: /usr/include/errno.h /usr/include/bits/errno.h
hash_map.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
hash_map.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
hash_map.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
hash_map.o: /usr/include/bits/time.h /usr/include/sys/select.h
hash_map.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
hash_map.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
hash_map.o: ../threads/rmutex_extension.h
hash_map.o: ../threads/rmutex_extension_abstract.h
hash_map.o: ../threads/auto_mutex_extension_abstract.h
hash_map.o: ../binary_search_tree.h
hash_map.o: ../binary_search_tree/binary_search_tree_kernel_1.h
hash_map.o: ../binary_search_tree/binary_search_tree_kernel_2.h
hash_map.o: ../binary_search_tree/binary_search_tree_kernel_c.h
hash_map.o: ../member_function_pointer.h
hash_map.o: ../member_function_pointer/member_function_pointer_kernel_1.h
hash_map.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
hash_map.o: ../member_function_pointer/member_function_pointer_kernel_c.h
hash_map.o: ../queue.h ../queue/queue_kernel_1.h
hash_map.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
hash_map.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
hash_map.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
hash_map.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
hash_map.o: ../set/set_kernel_c.h ../set/set_compare_1.h
hash_map.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
hash_map.o: ../threads/auto_unlock_extension.h
hash_map.o: ../threads/auto_unlock_extension_abstract.h
hash_map.o: ../threads/create_new_thread_extension.h
hash_map.o: ../threads/create_new_thread_extension_abstract.h
hash_map.o: ../threads/multithreaded_object_extension.h
hash_map.o: ../threads/multithreaded_object_extension_abstract.h
hash_map.o: ../threads/rsignaler_extension.h
hash_map.o: ../threads/rsignaler_extension_abstract.h ../map.h
hash_map.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
hash_map.o: ../threads/threaded_object_extension.h
hash_map.o: ../threads/threaded_object_extension_abstract.h
hash_map.o: ../threads/thread_specific_data_extension.h
hash_map.o: ../threads/thread_specific_data_extension_abstract.h
hash_map.o: ../threads/thread_function_extension.h
hash_map.o: ../threads/thread_function_extension_abstract.h
hash_map.o: ../threads/threaded_object_extension.h
hash_map.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
hash_map.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
hash_map.o: ../misc_api.h ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
hash_map.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
hash_map.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
hash_map.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
hash_map.o: ../smart_pointers/scoped_ptr_abstract.h
hash_map.o: ../smart_pointers/shared_ptr.h
hash_map.o: ../smart_pointers/shared_ptr_abstract.h
hash_map.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
hash_map.o: ../smart_pointers/weak_ptr_abstract.h
hash_map.o: ../../dlib/logger/extra_logger_headers.h
hash_map.o: ../../dlib/logger/logger_kernel_1.h
hash_map.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
hash_map.o: ../config_reader/config_reader_kernel_1.h
hash_map.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
hash_map.o: ../../dlib/map/map_kernel_1.h
hash_map.o: ../../dlib/map/map_kernel_abstract.h
hash_map.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
hash_map.o: ../tokenizer/tokenizer_kernel_1.h
hash_map.o: ../tokenizer/tokenizer_kernel_abstract.h
hash_map.o: ../tokenizer/tokenizer_kernel_c.h
hash_map.o: ../config_reader/config_reader_thread_safe_1.h
hash_map.o: ../config_reader/config_reader_thread_safe_abstract.h
hash_map.o: ../../dlib/assert.h ../../dlib/algs.h
hash_set.o: ../../dlib/hash_set.h ../../dlib/hash_set/hash_set_kernel_1.h
hash_set.o: ../../dlib/hash_set/hash_set_kernel_abstract.h ../algs.h
hash_set.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
hash_set.o: ../general_hash/general_hash.h ../interfaces/enumerable.h
hash_set.o: ../interfaces/remover.h ../assert.h ../serialize.h ../algs.h
hash_set.o: ../uintn.h ../interfaces/enumerable.h ../interfaces/map_pair.h
hash_set.o: ../enable_if.h ../memory_manager.h
hash_set.o: ../memory_manager/memory_manager_kernel_1.h
hash_set.o: ../memory_manager/memory_manager_kernel_abstract.h
hash_set.o: ../memory_manager/memory_manager_kernel_2.h
hash_set.o: ../memory_manager/memory_manager_kernel_3.h
hash_set.o: ../memory_manager/memory_manager_kernel_2.h
hash_set.o: ../binary_search_tree/binary_search_tree_kernel_2.h
hash_set.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
hash_set.o: ../interfaces/map_pair.h ../../dlib/hash_set/hash_set_kernel_c.h
hash_set.o: ../../dlib/hash_table.h
hash_set.o: ../../dlib/hash_table/hash_table_kernel_1.h
hash_set.o: ../../dlib/hash_table/hash_table_kernel_abstract.h
hash_set.o: ../../dlib/hash_table/hash_table_kernel_2.h
hash_set.o: ../../dlib/hash_table/hash_table_kernel_c.h
hash_set.o: ../../dlib/memory_manager.h binary_search_tree.h
hash_set.o: ../../dlib/memory_manager_global.h
hash_set.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
hash_set.o: ../memory_manager/memory_manager_kernel_abstract.h
hash_set.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
hash_set.o: ../../dlib/memory_manager_stateless.h
hash_set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
hash_set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
hash_set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
hash_set.o: ../threads.h ../threads/threads_kernel.h ../platform.h
hash_set.o: ../threads/posix.h ../threads/threads_kernel_2.h
hash_set.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
hash_set.o: /usr/include/features.h /usr/include/sys/cdefs.h
hash_set.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
hash_set.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
hash_set.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
hash_set.o: /usr/include/time.h /usr/include/bits/sched.h
hash_set.o: /usr/include/signal.h /usr/include/bits/sigset.h
hash_set.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
hash_set.o: /usr/include/errno.h /usr/include/bits/errno.h
hash_set.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
hash_set.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
hash_set.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
hash_set.o: /usr/include/bits/time.h /usr/include/sys/select.h
hash_set.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
hash_set.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
hash_set.o: ../threads/rmutex_extension.h
hash_set.o: ../threads/rmutex_extension_abstract.h
hash_set.o: ../threads/auto_mutex_extension_abstract.h
hash_set.o: ../binary_search_tree.h
hash_set.o: ../binary_search_tree/binary_search_tree_kernel_1.h
hash_set.o: ../binary_search_tree/binary_search_tree_kernel_2.h
hash_set.o: ../binary_search_tree/binary_search_tree_kernel_c.h
hash_set.o: ../member_function_pointer.h
hash_set.o: ../member_function_pointer/member_function_pointer_kernel_1.h
hash_set.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
hash_set.o: ../member_function_pointer/member_function_pointer_kernel_c.h
hash_set.o: ../queue.h ../queue/queue_kernel_1.h
hash_set.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
hash_set.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
hash_set.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
hash_set.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
hash_set.o: ../set/set_kernel_c.h ../set/set_compare_1.h
hash_set.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
hash_set.o: ../threads/auto_unlock_extension.h
hash_set.o: ../threads/auto_unlock_extension_abstract.h
hash_set.o: ../threads/create_new_thread_extension.h
hash_set.o: ../threads/create_new_thread_extension_abstract.h
hash_set.o: ../threads/multithreaded_object_extension.h
hash_set.o: ../threads/multithreaded_object_extension_abstract.h
hash_set.o: ../threads/rsignaler_extension.h
hash_set.o: ../threads/rsignaler_extension_abstract.h ../map.h
hash_set.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
hash_set.o: ../threads/threaded_object_extension.h
hash_set.o: ../threads/threaded_object_extension_abstract.h
hash_set.o: ../threads/thread_specific_data_extension.h
hash_set.o: ../threads/thread_specific_data_extension_abstract.h
hash_set.o: ../threads/thread_function_extension.h
hash_set.o: ../threads/thread_function_extension_abstract.h
hash_set.o: ../threads/threaded_object_extension.h
hash_set.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
hash_set.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
hash_set.o: ../misc_api.h ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
hash_set.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
hash_set.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
hash_set.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
hash_set.o: ../smart_pointers/scoped_ptr_abstract.h
hash_set.o: ../smart_pointers/shared_ptr.h
hash_set.o: ../smart_pointers/shared_ptr_abstract.h
hash_set.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
hash_set.o: ../smart_pointers/weak_ptr_abstract.h
hash_set.o: ../../dlib/logger/extra_logger_headers.h
hash_set.o: ../../dlib/logger/logger_kernel_1.h
hash_set.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
hash_set.o: ../config_reader/config_reader_kernel_1.h
hash_set.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
hash_set.o: ../../dlib/map/map_kernel_1.h
hash_set.o: ../../dlib/map/map_kernel_abstract.h
hash_set.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
hash_set.o: ../tokenizer/tokenizer_kernel_1.h
hash_set.o: ../tokenizer/tokenizer_kernel_abstract.h
hash_set.o: ../tokenizer/tokenizer_kernel_c.h
hash_set.o: ../config_reader/config_reader_thread_safe_1.h
hash_set.o: ../config_reader/config_reader_thread_safe_abstract.h
hash_set.o: ../../dlib/assert.h ../../dlib/algs.h
hash_table.o: ../../dlib/hash_table.h
hash_table.o: ../../dlib/hash_table/hash_table_kernel_1.h
hash_table.o: ../../dlib/hash_table/hash_table_kernel_abstract.h
hash_table.o: ../general_hash/general_hash.h ../algs.h ../platform.h
hash_table.o: ../assert.h ../error.h ../noncopyable.h
hash_table.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
hash_table.o: ../interfaces/remover.h ../assert.h ../serialize.h ../algs.h
hash_table.o: ../uintn.h ../interfaces/enumerable.h ../interfaces/map_pair.h
hash_table.o: ../enable_if.h ../memory_manager.h
hash_table.o: ../memory_manager/memory_manager_kernel_1.h
hash_table.o: ../memory_manager/memory_manager_kernel_abstract.h
hash_table.o: ../memory_manager/memory_manager_kernel_2.h
hash_table.o: ../memory_manager/memory_manager_kernel_3.h
hash_table.o: ../memory_manager/memory_manager_kernel_2.h
hash_table.o: ../binary_search_tree/binary_search_tree_kernel_2.h
hash_table.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
hash_table.o: ../../dlib/hash_table/hash_table_kernel_2.h
hash_table.o: ../../dlib/hash_table/hash_table_kernel_c.h
hash_table.o: ../../dlib/memory_manager.h binary_search_tree.h
hash_table.o: ../../dlib/memory_manager_global.h
hash_table.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
hash_table.o: ../memory_manager/memory_manager_kernel_abstract.h
hash_table.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
hash_table.o: ../../dlib/memory_manager_stateless.h
hash_table.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
hash_table.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
hash_table.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
hash_table.o: ../threads.h ../threads/threads_kernel.h ../platform.h
hash_table.o: ../threads/posix.h ../threads/threads_kernel_2.h
hash_table.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
hash_table.o: /usr/include/features.h /usr/include/sys/cdefs.h
hash_table.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
hash_table.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
hash_table.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
hash_table.o: /usr/include/time.h /usr/include/bits/sched.h
hash_table.o: /usr/include/signal.h /usr/include/bits/sigset.h
hash_table.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
hash_table.o: /usr/include/errno.h /usr/include/bits/errno.h
hash_table.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
hash_table.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
hash_table.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
hash_table.o: /usr/include/bits/time.h /usr/include/sys/select.h
hash_table.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
hash_table.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
hash_table.o: ../threads/rmutex_extension.h
hash_table.o: ../threads/rmutex_extension_abstract.h
hash_table.o: ../threads/auto_mutex_extension_abstract.h
hash_table.o: ../binary_search_tree.h
hash_table.o: ../binary_search_tree/binary_search_tree_kernel_1.h
hash_table.o: ../binary_search_tree/binary_search_tree_kernel_2.h
hash_table.o: ../binary_search_tree/binary_search_tree_kernel_c.h
hash_table.o: ../member_function_pointer.h
hash_table.o: ../member_function_pointer/member_function_pointer_kernel_1.h
hash_table.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
hash_table.o: ../member_function_pointer/member_function_pointer_kernel_c.h
hash_table.o: ../queue.h ../queue/queue_kernel_1.h
hash_table.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
hash_table.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
hash_table.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
hash_table.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
hash_table.o: ../set/set_kernel_c.h ../set/set_compare_1.h
hash_table.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
hash_table.o: ../threads/auto_unlock_extension.h
hash_table.o: ../threads/auto_unlock_extension_abstract.h
hash_table.o: ../threads/create_new_thread_extension.h
hash_table.o: ../threads/create_new_thread_extension_abstract.h
hash_table.o: ../threads/multithreaded_object_extension.h
hash_table.o: ../threads/multithreaded_object_extension_abstract.h
hash_table.o: ../threads/rsignaler_extension.h
hash_table.o: ../threads/rsignaler_extension_abstract.h ../map.h
hash_table.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
hash_table.o: ../threads/threaded_object_extension.h
hash_table.o: ../threads/threaded_object_extension_abstract.h
hash_table.o: ../threads/thread_specific_data_extension.h
hash_table.o: ../threads/thread_specific_data_extension_abstract.h
hash_table.o: ../threads/thread_function_extension.h
hash_table.o: ../threads/thread_function_extension_abstract.h
hash_table.o: ../threads/threaded_object_extension.h
hash_table.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
hash_table.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
hash_table.o: ../misc_api.h ../misc_api/posix.h
hash_table.o: ../misc_api/misc_api_kernel_2.h
hash_table.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
hash_table.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
hash_table.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
hash_table.o: ../smart_pointers/scoped_ptr_abstract.h
hash_table.o: ../smart_pointers/shared_ptr.h
hash_table.o: ../smart_pointers/shared_ptr_abstract.h
hash_table.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
hash_table.o: ../smart_pointers/weak_ptr_abstract.h
hash_table.o: ../../dlib/logger/extra_logger_headers.h
hash_table.o: ../../dlib/logger/logger_kernel_1.h
hash_table.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
hash_table.o: ../config_reader/config_reader_kernel_1.h
hash_table.o: ../config_reader/config_reader_kernel_abstract.h
hash_table.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
hash_table.o: ../../dlib/map/map_kernel_abstract.h
hash_table.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
hash_table.o: ../tokenizer/tokenizer_kernel_1.h
hash_table.o: ../tokenizer/tokenizer_kernel_abstract.h
hash_table.o: ../tokenizer/tokenizer_kernel_c.h
hash_table.o: ../config_reader/config_reader_thread_safe_1.h
hash_table.o: ../config_reader/config_reader_thread_safe_abstract.h
hash_table.o: ../../dlib/assert.h ../../dlib/algs.h
image.o: ../../dlib/pixel.h ../serialize.h ../algs.h ../uintn.h
image.o: ../enable_if.h ../../dlib/array2d.h
image.o: ../../dlib/array2d/array2d_kernel_1.h
image.o: ../../dlib/array2d/array2d_kernel_abstract.h ../algs.h ../platform.h
image.o: ../assert.h ../error.h ../noncopyable.h ../interfaces/enumerable.h
image.o: ../serialize.h ../interfaces/enumerable.h ../interfaces/map_pair.h
image.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
image.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
image.o: ../memory_manager/memory_manager_kernel_2.h
image.o: ../memory_manager/memory_manager_kernel_3.h
image.o: ../memory_manager/memory_manager_kernel_2.h
image.o: ../binary_search_tree/binary_search_tree_kernel_2.h
image.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
image.o: ../interfaces/map_pair.h ../interfaces/remover.h
image.o: ../../dlib/array2d/array2d_kernel_c.h ../../dlib/memory_manager.h
image.o: ../../dlib/image_transforms.h
image.o: ../../dlib/image_transforms/assign_image.h ../pixel.h
image.o: ../../dlib/image_transforms/assign_image_abstract.h
image.o: ../../dlib/image_transforms/equalize_histogram.h
image.o: ../../dlib/image_transforms/equalize_histogram_abstract.h
image.o: ../enable_if.h ../matrix.h ../matrix/matrix.h
image.o: ../matrix/matrix_abstract.h ../matrix/matrix_utilities.h
image.o: ../matrix/matrix_utilities_abstract.h ../matrix/matrix.h
image.o: ../matrix/matrix_math_functions.h ../matrix/matrix_utilities.h
image.o: ../../dlib/image_transforms/morphological_operations.h
image.o: ../../dlib/image_transforms/thresholding.h
image.o: ../../dlib/image_transforms/thresholding_abstract.h
image.o: ../../dlib/image_transforms/equalize_histogram.h
image.o: ../../dlib/image_transforms/morphological_operations_abstract.h
image.o: ../../dlib/image_transforms/spatial_filtering.h
image.o: ../../dlib/image_transforms/spatial_filtering_abstract.h
image.o: ../../dlib/image_transforms/thresholding.h
image.o: ../../dlib/image_transforms/edge_detector.h
image.o: ../../dlib/image_transforms/edge_detector_abstract.h ../array2d.h
image.o: tester.h ../../dlib/map.h ../../dlib/logger.h
image.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
image.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
image.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
image.o: /usr/include/pthread.h /usr/include/features.h
image.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
image.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
image.o: /usr/include/sched.h /usr/include/bits/types.h
image.o: /usr/include/bits/typesizes.h /usr/include/time.h
image.o: /usr/include/bits/sched.h /usr/include/signal.h
image.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
image.o: /usr/include/bits/setjmp.h /usr/include/errno.h
image.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
image.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
image.o: /usr/include/asm-generic/errno.h
image.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
image.o: /usr/include/bits/time.h /usr/include/sys/select.h
image.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
image.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
image.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
image.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
image.o: ../binary_search_tree/binary_search_tree_kernel_1.h
image.o: ../binary_search_tree/binary_search_tree_kernel_2.h
image.o: ../binary_search_tree/binary_search_tree_kernel_c.h
image.o: ../member_function_pointer.h
image.o: ../member_function_pointer/member_function_pointer_kernel_1.h
image.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
image.o: ../member_function_pointer/member_function_pointer_kernel_c.h
image.o: ../queue.h ../queue/queue_kernel_1.h
image.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
image.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
image.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
image.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
image.o: ../set/set_kernel_c.h binary_search_tree.h
image.o: ../../dlib/memory_manager_global.h
image.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
image.o: ../memory_manager/memory_manager_kernel_abstract.h
image.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
image.o: ../../dlib/memory_manager_stateless.h
image.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
image.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
image.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
image.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
image.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
image.o: ../threads/auto_unlock_extension.h
image.o: ../threads/auto_unlock_extension_abstract.h
image.o: ../threads/create_new_thread_extension.h
image.o: ../threads/create_new_thread_extension_abstract.h
image.o: ../threads/multithreaded_object_extension.h
image.o: ../threads/multithreaded_object_extension_abstract.h
image.o: ../threads/rsignaler_extension.h
image.o: ../threads/rsignaler_extension_abstract.h ../map.h
image.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
image.o: ../threads/threaded_object_extension.h
image.o: ../threads/threaded_object_extension_abstract.h
image.o: ../threads/thread_specific_data_extension.h
image.o: ../threads/thread_specific_data_extension_abstract.h
image.o: ../threads/thread_function_extension.h
image.o: ../threads/thread_function_extension_abstract.h
image.o: ../threads/threaded_object_extension.h ../misc_api.h
image.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
image.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
image.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
image.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
image.o: ../smart_pointers/scoped_ptr_abstract.h
image.o: ../smart_pointers/shared_ptr.h
image.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
image.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
image.o: ../../dlib/logger/extra_logger_headers.h
image.o: ../../dlib/logger/logger_kernel_1.h
image.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
image.o: ../config_reader/config_reader_kernel_1.h
image.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
image.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
image.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
image.o: ../tokenizer/tokenizer_kernel_1.h
image.o: ../tokenizer/tokenizer_kernel_abstract.h
image.o: ../tokenizer/tokenizer_kernel_c.h
image.o: ../config_reader/config_reader_thread_safe_1.h
image.o: ../config_reader/config_reader_thread_safe_abstract.h
image.o: ../../dlib/assert.h ../../dlib/algs.h
lz77_buffer.o: ../../dlib/sliding_buffer.h
lz77_buffer.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_1.h
lz77_buffer.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_abstract.h
lz77_buffer.o: ../algs.h ../platform.h ../assert.h ../error.h
lz77_buffer.o: ../noncopyable.h ../interfaces/enumerable.h ../serialize.h
lz77_buffer.o: ../algs.h ../uintn.h ../interfaces/enumerable.h
lz77_buffer.o: ../interfaces/map_pair.h ../enable_if.h
lz77_buffer.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_c.h
lz77_buffer.o: ../assert.h ../../dlib/lz77_buffer.h
lz77_buffer.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_1.h
lz77_buffer.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_abstract.h
lz77_buffer.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_2.h
lz77_buffer.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_c.h
lz77_buffer.o: ../../dlib/sliding_buffer.h tester.h ../../dlib/map.h
lz77_buffer.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
lz77_buffer.o: ../threads.h ../threads/threads_kernel.h ../platform.h
lz77_buffer.o: ../threads/posix.h ../threads/threads_kernel_2.h
lz77_buffer.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
lz77_buffer.o: /usr/include/features.h /usr/include/sys/cdefs.h
lz77_buffer.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
lz77_buffer.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
lz77_buffer.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
lz77_buffer.o: /usr/include/time.h /usr/include/bits/sched.h
lz77_buffer.o: /usr/include/signal.h /usr/include/bits/sigset.h
lz77_buffer.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
lz77_buffer.o: /usr/include/errno.h /usr/include/bits/errno.h
lz77_buffer.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
lz77_buffer.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
lz77_buffer.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
lz77_buffer.o: /usr/include/bits/time.h /usr/include/sys/select.h
lz77_buffer.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
lz77_buffer.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
lz77_buffer.o: ../threads/rmutex_extension.h
lz77_buffer.o: ../threads/rmutex_extension_abstract.h
lz77_buffer.o: ../threads/auto_mutex_extension_abstract.h
lz77_buffer.o: ../binary_search_tree.h
lz77_buffer.o: ../binary_search_tree/binary_search_tree_kernel_1.h
lz77_buffer.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
lz77_buffer.o: ../interfaces/map_pair.h ../interfaces/remover.h
lz77_buffer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
lz77_buffer.o: ../binary_search_tree/binary_search_tree_kernel_c.h
lz77_buffer.o: ../../dlib/memory_manager.h ../member_function_pointer.h
lz77_buffer.o: ../member_function_pointer/member_function_pointer_kernel_1.h
lz77_buffer.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
lz77_buffer.o: ../member_function_pointer/member_function_pointer_kernel_c.h
lz77_buffer.o: ../memory_manager.h
lz77_buffer.o: ../memory_manager/memory_manager_kernel_1.h
lz77_buffer.o: ../memory_manager/memory_manager_kernel_abstract.h
lz77_buffer.o: ../memory_manager/memory_manager_kernel_2.h
lz77_buffer.o: ../memory_manager/memory_manager_kernel_3.h
lz77_buffer.o: ../memory_manager/memory_manager_kernel_2.h
lz77_buffer.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
lz77_buffer.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
lz77_buffer.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
lz77_buffer.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h
lz77_buffer.o: ../sort.h ../set.h ../set/set_kernel_1.h
lz77_buffer.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
lz77_buffer.o: binary_search_tree.h ../../dlib/memory_manager_global.h
lz77_buffer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
lz77_buffer.o: ../memory_manager/memory_manager_kernel_abstract.h
lz77_buffer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
lz77_buffer.o: ../../dlib/memory_manager_stateless.h
lz77_buffer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
lz77_buffer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
lz77_buffer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
lz77_buffer.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
lz77_buffer.o: ../set/set_compare_abstract.h
lz77_buffer.o: ../threads/auto_mutex_extension.h
lz77_buffer.o: ../threads/auto_unlock_extension.h
lz77_buffer.o: ../threads/auto_unlock_extension_abstract.h
lz77_buffer.o: ../threads/create_new_thread_extension.h
lz77_buffer.o: ../threads/create_new_thread_extension_abstract.h
lz77_buffer.o: ../threads/multithreaded_object_extension.h
lz77_buffer.o: ../threads/multithreaded_object_extension_abstract.h
lz77_buffer.o: ../threads/rsignaler_extension.h
lz77_buffer.o: ../threads/rsignaler_extension_abstract.h ../map.h
lz77_buffer.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
lz77_buffer.o: ../threads/threaded_object_extension.h
lz77_buffer.o: ../threads/threaded_object_extension_abstract.h
lz77_buffer.o: ../threads/thread_specific_data_extension.h
lz77_buffer.o: ../threads/thread_specific_data_extension_abstract.h
lz77_buffer.o: ../threads/thread_function_extension.h
lz77_buffer.o: ../threads/thread_function_extension_abstract.h
lz77_buffer.o: ../threads/threaded_object_extension.h ../misc_api.h
lz77_buffer.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
lz77_buffer.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
lz77_buffer.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
lz77_buffer.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
lz77_buffer.o: ../smart_pointers/scoped_ptr_abstract.h
lz77_buffer.o: ../smart_pointers/shared_ptr.h
lz77_buffer.o: ../smart_pointers/shared_ptr_abstract.h
lz77_buffer.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
lz77_buffer.o: ../smart_pointers/weak_ptr_abstract.h
lz77_buffer.o: ../../dlib/logger/extra_logger_headers.h
lz77_buffer.o: ../../dlib/logger/logger_kernel_1.h
lz77_buffer.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
lz77_buffer.o: ../config_reader/config_reader_kernel_1.h
lz77_buffer.o: ../config_reader/config_reader_kernel_abstract.h
lz77_buffer.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
lz77_buffer.o: ../../dlib/map/map_kernel_abstract.h
lz77_buffer.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
lz77_buffer.o: ../tokenizer/tokenizer_kernel_1.h
lz77_buffer.o: ../tokenizer/tokenizer_kernel_abstract.h
lz77_buffer.o: ../tokenizer/tokenizer_kernel_c.h
lz77_buffer.o: ../config_reader/config_reader_thread_safe_1.h
lz77_buffer.o: ../config_reader/config_reader_thread_safe_abstract.h
lz77_buffer.o: ../../dlib/assert.h ../../dlib/algs.h
map.o: ../../dlib/map.h tester.h ../../dlib/logger.h
map.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
map.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
map.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
map.o: /usr/include/pthread.h /usr/include/features.h
map.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
map.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
map.o: /usr/include/sched.h /usr/include/bits/types.h
map.o: /usr/include/bits/typesizes.h /usr/include/time.h
map.o: /usr/include/bits/sched.h /usr/include/signal.h
map.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
map.o: /usr/include/bits/setjmp.h /usr/include/errno.h
map.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
map.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
map.o: /usr/include/asm-generic/errno.h /usr/include/asm-generic/errno-base.h
map.o: /usr/include/sys/time.h /usr/include/bits/time.h
map.o: /usr/include/sys/select.h /usr/include/bits/select.h ../algs.h
map.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
map.o: ../threads/threads_kernel_shared.h ../threads/auto_mutex_extension.h
map.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
map.o: ../threads/rmutex_extension_abstract.h
map.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
map.o: ../binary_search_tree/binary_search_tree_kernel_1.h
map.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
map.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
map.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
map.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
map.o: ../binary_search_tree/binary_search_tree_kernel_2.h
map.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
map.o: ../../dlib/memory_manager.h ../member_function_pointer.h
map.o: ../member_function_pointer/member_function_pointer_kernel_1.h
map.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
map.o: ../member_function_pointer/member_function_pointer_kernel_c.h
map.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
map.o: ../memory_manager/memory_manager_kernel_abstract.h
map.o: ../memory_manager/memory_manager_kernel_2.h
map.o: ../memory_manager/memory_manager_kernel_3.h
map.o: ../memory_manager/memory_manager_kernel_2.h
map.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
map.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
map.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
map.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
map.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
map.o: ../set/set_kernel_c.h binary_search_tree.h
map.o: ../../dlib/memory_manager_global.h
map.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
map.o: ../memory_manager/memory_manager_kernel_abstract.h
map.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
map.o: ../../dlib/memory_manager_stateless.h
map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
map.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
map.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
map.o: ../threads/auto_unlock_extension.h
map.o: ../threads/auto_unlock_extension_abstract.h
map.o: ../threads/create_new_thread_extension.h
map.o: ../threads/create_new_thread_extension_abstract.h
map.o: ../threads/multithreaded_object_extension.h
map.o: ../threads/multithreaded_object_extension_abstract.h
map.o: ../threads/rsignaler_extension.h
map.o: ../threads/rsignaler_extension_abstract.h ../map.h
map.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
map.o: ../threads/threaded_object_extension.h
map.o: ../threads/threaded_object_extension_abstract.h
map.o: ../threads/thread_specific_data_extension.h
map.o: ../threads/thread_specific_data_extension_abstract.h
map.o: ../threads/thread_function_extension.h
map.o: ../threads/thread_function_extension_abstract.h
map.o: ../threads/threaded_object_extension.h ../misc_api.h
map.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
map.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
map.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
map.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
map.o: ../smart_pointers/scoped_ptr_abstract.h ../smart_pointers/shared_ptr.h
map.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
map.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
map.o: ../../dlib/logger/extra_logger_headers.h
map.o: ../../dlib/logger/logger_kernel_1.h
map.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
map.o: ../config_reader/config_reader_kernel_1.h
map.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
map.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
map.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
map.o: ../tokenizer/tokenizer_kernel_1.h
map.o: ../tokenizer/tokenizer_kernel_abstract.h
map.o: ../tokenizer/tokenizer_kernel_c.h
map.o: ../config_reader/config_reader_thread_safe_1.h
map.o: ../config_reader/config_reader_thread_safe_abstract.h
map.o: ../../dlib/assert.h ../../dlib/algs.h
matrix.o: ../../dlib/matrix.h ../matrix/matrix.h ../matrix/matrix_abstract.h
matrix.o: ../algs.h ../platform.h ../assert.h ../error.h ../noncopyable.h
matrix.o: ../serialize.h ../algs.h ../uintn.h ../interfaces/enumerable.h
matrix.o: ../interfaces/map_pair.h ../enable_if.h ../enable_if.h
matrix.o: ../matrix/matrix_utilities.h ../matrix/matrix_utilities_abstract.h
matrix.o: ../matrix/matrix.h ../pixel.h ../serialize.h
matrix.o: ../matrix/matrix_math_functions.h ../matrix/matrix_utilities.h
matrix.o: tester.h ../../dlib/map.h ../../dlib/logger.h
matrix.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
matrix.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
matrix.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
matrix.o: /usr/include/pthread.h /usr/include/features.h
matrix.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
matrix.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
matrix.o: /usr/include/sched.h /usr/include/bits/types.h
matrix.o: /usr/include/bits/typesizes.h /usr/include/time.h
matrix.o: /usr/include/bits/sched.h /usr/include/signal.h
matrix.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
matrix.o: /usr/include/bits/setjmp.h /usr/include/errno.h
matrix.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
matrix.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
matrix.o: /usr/include/asm-generic/errno.h
matrix.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
matrix.o: /usr/include/bits/time.h /usr/include/sys/select.h
matrix.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
matrix.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
matrix.o: ../threads/rmutex_extension.h
matrix.o: ../threads/rmutex_extension_abstract.h
matrix.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
matrix.o: ../binary_search_tree/binary_search_tree_kernel_1.h
matrix.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
matrix.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
matrix.o: ../interfaces/remover.h
matrix.o: ../binary_search_tree/binary_search_tree_kernel_2.h
matrix.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
matrix.o: ../../dlib/memory_manager.h ../member_function_pointer.h
matrix.o: ../member_function_pointer/member_function_pointer_kernel_1.h
matrix.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
matrix.o: ../member_function_pointer/member_function_pointer_kernel_c.h
matrix.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
matrix.o: ../memory_manager/memory_manager_kernel_abstract.h
matrix.o: ../memory_manager/memory_manager_kernel_2.h
matrix.o: ../memory_manager/memory_manager_kernel_3.h
matrix.o: ../memory_manager/memory_manager_kernel_2.h
matrix.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
matrix.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
matrix.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
matrix.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
matrix.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
matrix.o: ../set/set_kernel_c.h binary_search_tree.h
matrix.o: ../../dlib/memory_manager_global.h
matrix.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
matrix.o: ../memory_manager/memory_manager_kernel_abstract.h
matrix.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
matrix.o: ../../dlib/memory_manager_stateless.h
matrix.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
matrix.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
matrix.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
matrix.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
matrix.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
matrix.o: ../threads/auto_unlock_extension.h
matrix.o: ../threads/auto_unlock_extension_abstract.h
matrix.o: ../threads/create_new_thread_extension.h
matrix.o: ../threads/create_new_thread_extension_abstract.h
matrix.o: ../threads/multithreaded_object_extension.h
matrix.o: ../threads/multithreaded_object_extension_abstract.h
matrix.o: ../threads/rsignaler_extension.h
matrix.o: ../threads/rsignaler_extension_abstract.h ../map.h
matrix.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
matrix.o: ../threads/threaded_object_extension.h
matrix.o: ../threads/threaded_object_extension_abstract.h
matrix.o: ../threads/thread_specific_data_extension.h
matrix.o: ../threads/thread_specific_data_extension_abstract.h
matrix.o: ../threads/thread_function_extension.h
matrix.o: ../threads/thread_function_extension_abstract.h
matrix.o: ../threads/threaded_object_extension.h ../misc_api.h
matrix.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
matrix.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
matrix.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
matrix.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
matrix.o: ../smart_pointers/scoped_ptr_abstract.h
matrix.o: ../smart_pointers/shared_ptr.h
matrix.o: ../smart_pointers/shared_ptr_abstract.h
matrix.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
matrix.o: ../smart_pointers/weak_ptr_abstract.h
matrix.o: ../../dlib/logger/extra_logger_headers.h
matrix.o: ../../dlib/logger/logger_kernel_1.h
matrix.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
matrix.o: ../config_reader/config_reader_kernel_1.h
matrix.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
matrix.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
matrix.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
matrix.o: ../tokenizer/tokenizer_kernel_1.h
matrix.o: ../tokenizer/tokenizer_kernel_abstract.h
matrix.o: ../tokenizer/tokenizer_kernel_c.h
matrix.o: ../config_reader/config_reader_thread_safe_1.h
matrix.o: ../config_reader/config_reader_thread_safe_abstract.h
matrix.o: ../../dlib/assert.h ../../dlib/algs.h
md5.o: ../../dlib/md5.h ../../dlib/md5/md5_kernel_1.h
md5.o: ../md5/md5_kernel_abstract.h ../algs.h ../platform.h ../assert.h
md5.o: ../error.h ../noncopyable.h tester.h ../../dlib/map.h
md5.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../threads.h
md5.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
md5.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
md5.o: /usr/include/pthread.h /usr/include/features.h
md5.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
md5.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
md5.o: /usr/include/sched.h /usr/include/bits/types.h
md5.o: /usr/include/bits/typesizes.h /usr/include/time.h
md5.o: /usr/include/bits/sched.h /usr/include/signal.h
md5.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
md5.o: /usr/include/bits/setjmp.h /usr/include/errno.h
md5.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
md5.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
md5.o: /usr/include/asm-generic/errno.h /usr/include/asm-generic/errno-base.h
md5.o: /usr/include/sys/time.h /usr/include/bits/time.h
md5.o: /usr/include/sys/select.h /usr/include/bits/select.h
md5.o: ../threads/threads_kernel_shared.h ../threads/auto_mutex_extension.h
md5.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
md5.o: ../threads/rmutex_extension_abstract.h
md5.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
md5.o: ../binary_search_tree/binary_search_tree_kernel_1.h
md5.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
md5.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
md5.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
md5.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
md5.o: ../binary_search_tree/binary_search_tree_kernel_2.h
md5.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
md5.o: ../../dlib/memory_manager.h ../member_function_pointer.h
md5.o: ../member_function_pointer/member_function_pointer_kernel_1.h
md5.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
md5.o: ../member_function_pointer/member_function_pointer_kernel_c.h
md5.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
md5.o: ../memory_manager/memory_manager_kernel_abstract.h
md5.o: ../memory_manager/memory_manager_kernel_2.h
md5.o: ../memory_manager/memory_manager_kernel_3.h
md5.o: ../memory_manager/memory_manager_kernel_2.h
md5.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
md5.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
md5.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
md5.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
md5.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
md5.o: ../set/set_kernel_c.h binary_search_tree.h
md5.o: ../../dlib/memory_manager_global.h
md5.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
md5.o: ../memory_manager/memory_manager_kernel_abstract.h
md5.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
md5.o: ../../dlib/memory_manager_stateless.h
md5.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
md5.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
md5.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
md5.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
md5.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
md5.o: ../threads/auto_unlock_extension.h
md5.o: ../threads/auto_unlock_extension_abstract.h
md5.o: ../threads/create_new_thread_extension.h
md5.o: ../threads/create_new_thread_extension_abstract.h
md5.o: ../threads/multithreaded_object_extension.h
md5.o: ../threads/multithreaded_object_extension_abstract.h
md5.o: ../threads/rsignaler_extension.h
md5.o: ../threads/rsignaler_extension_abstract.h ../map.h
md5.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
md5.o: ../threads/threaded_object_extension.h
md5.o: ../threads/threaded_object_extension_abstract.h
md5.o: ../threads/thread_specific_data_extension.h
md5.o: ../threads/thread_specific_data_extension_abstract.h
md5.o: ../threads/thread_function_extension.h
md5.o: ../threads/thread_function_extension_abstract.h
md5.o: ../threads/threaded_object_extension.h ../misc_api.h
md5.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
md5.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
md5.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
md5.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
md5.o: ../smart_pointers/scoped_ptr_abstract.h ../smart_pointers/shared_ptr.h
md5.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
md5.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
md5.o: ../../dlib/logger/extra_logger_headers.h
md5.o: ../../dlib/logger/logger_kernel_1.h
md5.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
md5.o: ../config_reader/config_reader_kernel_1.h
md5.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
md5.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
md5.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
md5.o: ../tokenizer/tokenizer_kernel_1.h
md5.o: ../tokenizer/tokenizer_kernel_abstract.h
md5.o: ../tokenizer/tokenizer_kernel_c.h
md5.o: ../config_reader/config_reader_thread_safe_1.h
md5.o: ../config_reader/config_reader_thread_safe_abstract.h
md5.o: ../../dlib/assert.h ../../dlib/algs.h
member_function_pointer.o: ../../dlib/member_function_pointer.h
member_function_pointer.o: ../member_function_pointer/member_function_pointer_kernel_1.h
member_function_pointer.o: ../algs.h ../platform.h ../assert.h ../error.h
member_function_pointer.o: ../noncopyable.h
member_function_pointer.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
member_function_pointer.o: ../member_function_pointer/member_function_pointer_kernel_c.h
member_function_pointer.o: ../assert.h tester.h ../../dlib/map.h
member_function_pointer.o: ../../dlib/logger.h
member_function_pointer.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
member_function_pointer.o: ../threads/threads_kernel.h ../platform.h
member_function_pointer.o: ../threads/posix.h ../threads/threads_kernel_2.h
member_function_pointer.o: ../threads/threads_kernel_abstract.h
member_function_pointer.o: /usr/include/pthread.h /usr/include/features.h
member_function_pointer.o: /usr/include/sys/cdefs.h
member_function_pointer.o: /usr/include/bits/wordsize.h
member_function_pointer.o: /usr/include/gnu/stubs.h
member_function_pointer.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
member_function_pointer.o: /usr/include/bits/types.h
member_function_pointer.o: /usr/include/bits/typesizes.h /usr/include/time.h
member_function_pointer.o: /usr/include/bits/sched.h /usr/include/signal.h
member_function_pointer.o: /usr/include/bits/sigset.h
member_function_pointer.o: /usr/include/bits/pthreadtypes.h
member_function_pointer.o: /usr/include/bits/setjmp.h /usr/include/errno.h
member_function_pointer.o: /usr/include/bits/errno.h
member_function_pointer.o: /usr/include/linux/errno.h
member_function_pointer.o: /usr/include/asm/errno.h
member_function_pointer.o: /usr/include/asm-i386/errno.h
member_function_pointer.o: /usr/include/asm-generic/errno.h
member_function_pointer.o: /usr/include/asm-generic/errno-base.h
member_function_pointer.o: /usr/include/sys/time.h /usr/include/bits/time.h
member_function_pointer.o: /usr/include/sys/select.h
member_function_pointer.o: /usr/include/bits/select.h
member_function_pointer.o: ../threads/threads_kernel_shared.h
member_function_pointer.o: ../threads/auto_mutex_extension.h
member_function_pointer.o: ../threads/threads_kernel.h
member_function_pointer.o: ../threads/rmutex_extension.h
member_function_pointer.o: ../threads/rmutex_extension_abstract.h
member_function_pointer.o: ../threads/auto_mutex_extension_abstract.h
member_function_pointer.o: ../binary_search_tree.h
member_function_pointer.o: ../binary_search_tree/binary_search_tree_kernel_1.h
member_function_pointer.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
member_function_pointer.o: ../interfaces/map_pair.h
member_function_pointer.o: ../interfaces/enumerable.h ../interfaces/remover.h
member_function_pointer.o: ../serialize.h ../algs.h ../uintn.h
member_function_pointer.o: ../interfaces/enumerable.h
member_function_pointer.o: ../interfaces/map_pair.h ../enable_if.h
member_function_pointer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
member_function_pointer.o: ../binary_search_tree/binary_search_tree_kernel_c.h
member_function_pointer.o: ../../dlib/memory_manager.h
member_function_pointer.o: ../member_function_pointer.h ../memory_manager.h
member_function_pointer.o: ../memory_manager/memory_manager_kernel_1.h
member_function_pointer.o: ../memory_manager/memory_manager_kernel_abstract.h
member_function_pointer.o: ../memory_manager/memory_manager_kernel_2.h
member_function_pointer.o: ../memory_manager/memory_manager_kernel_3.h
member_function_pointer.o: ../memory_manager/memory_manager_kernel_2.h
member_function_pointer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
member_function_pointer.o: ../queue.h ../queue/queue_kernel_1.h
member_function_pointer.o: ../queue/queue_kernel_abstract.h
member_function_pointer.o: ../queue/queue_kernel_2.h
member_function_pointer.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
member_function_pointer.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
member_function_pointer.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
member_function_pointer.o: ../set/set_kernel_c.h binary_search_tree.h
member_function_pointer.o: ../../dlib/memory_manager_global.h
member_function_pointer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
member_function_pointer.o: ../memory_manager/memory_manager_kernel_abstract.h
member_function_pointer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
member_function_pointer.o: ../../dlib/memory_manager_stateless.h
member_function_pointer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
member_function_pointer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
member_function_pointer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
member_function_pointer.o: ../../dlib/binary_search_tree.h
member_function_pointer.o: ../set/set_compare_1.h
member_function_pointer.o: ../set/set_compare_abstract.h
member_function_pointer.o: ../threads/auto_mutex_extension.h
member_function_pointer.o: ../threads/auto_unlock_extension.h
member_function_pointer.o: ../threads/auto_unlock_extension_abstract.h
member_function_pointer.o: ../threads/create_new_thread_extension.h
member_function_pointer.o: ../threads/create_new_thread_extension_abstract.h
member_function_pointer.o: ../threads/multithreaded_object_extension.h
member_function_pointer.o: ../threads/multithreaded_object_extension_abstract.h
member_function_pointer.o: ../threads/rsignaler_extension.h
member_function_pointer.o: ../threads/rsignaler_extension_abstract.h ../map.h
member_function_pointer.o: ../threads/rmutex_extension.h
member_function_pointer.o: ../threads/rsignaler_extension.h
member_function_pointer.o: ../threads/threaded_object_extension.h
member_function_pointer.o: ../threads/threaded_object_extension_abstract.h
member_function_pointer.o: ../threads/thread_specific_data_extension.h
member_function_pointer.o: ../threads/thread_specific_data_extension_abstract.h
member_function_pointer.o: ../threads/thread_function_extension.h
member_function_pointer.o: ../threads/thread_function_extension_abstract.h
member_function_pointer.o: ../threads/threaded_object_extension.h
member_function_pointer.o: ../misc_api.h ../misc_api/posix.h
member_function_pointer.o: ../misc_api/misc_api_kernel_2.h
member_function_pointer.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
member_function_pointer.o: ../../dlib/logger/logger_kernel_abstract.h
member_function_pointer.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
member_function_pointer.o: ../noncopyable.h
member_function_pointer.o: ../smart_pointers/scoped_ptr_abstract.h
member_function_pointer.o: ../smart_pointers/shared_ptr.h
member_function_pointer.o: ../smart_pointers/shared_ptr_abstract.h
member_function_pointer.o: ../smart_pointers/weak_ptr.h
member_function_pointer.o: ../smart_pointers/shared_ptr.h
member_function_pointer.o: ../smart_pointers/weak_ptr_abstract.h
member_function_pointer.o: ../../dlib/logger/extra_logger_headers.h
member_function_pointer.o: ../../dlib/logger/logger_kernel_1.h
member_function_pointer.o: ../../dlib/logger/logger_config_file.h
member_function_pointer.o: ../config_reader.h
member_function_pointer.o: ../config_reader/config_reader_kernel_1.h
member_function_pointer.o: ../config_reader/config_reader_kernel_abstract.h
member_function_pointer.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
member_function_pointer.o: ../../dlib/map/map_kernel_abstract.h
member_function_pointer.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
member_function_pointer.o: ../tokenizer/tokenizer_kernel_1.h
member_function_pointer.o: ../tokenizer/tokenizer_kernel_abstract.h
member_function_pointer.o: ../tokenizer/tokenizer_kernel_c.h
member_function_pointer.o: ../config_reader/config_reader_thread_safe_1.h
member_function_pointer.o: ../config_reader/config_reader_thread_safe_abstract.h
member_function_pointer.o: ../../dlib/assert.h ../../dlib/algs.h
metaprogramming.o: ../../dlib/algs.h tester.h ../../dlib/map.h
metaprogramming.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
metaprogramming.o: ../threads.h ../threads/threads_kernel.h ../platform.h
metaprogramming.o: ../threads/posix.h ../threads/threads_kernel_2.h
metaprogramming.o: ../threads/threads_kernel_abstract.h
metaprogramming.o: /usr/include/pthread.h /usr/include/features.h
metaprogramming.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
metaprogramming.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
metaprogramming.o: /usr/include/sched.h /usr/include/bits/types.h
metaprogramming.o: /usr/include/bits/typesizes.h /usr/include/time.h
metaprogramming.o: /usr/include/bits/sched.h /usr/include/signal.h
metaprogramming.o: /usr/include/bits/sigset.h
metaprogramming.o: /usr/include/bits/pthreadtypes.h
metaprogramming.o: /usr/include/bits/setjmp.h /usr/include/errno.h
metaprogramming.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
metaprogramming.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
metaprogramming.o: /usr/include/asm-generic/errno.h
metaprogramming.o: /usr/include/asm-generic/errno-base.h
metaprogramming.o: /usr/include/sys/time.h /usr/include/bits/time.h
metaprogramming.o: /usr/include/sys/select.h /usr/include/bits/select.h
metaprogramming.o: ../algs.h ../platform.h ../assert.h ../error.h
metaprogramming.o: ../noncopyable.h ../threads/threads_kernel_shared.h
metaprogramming.o: ../threads/auto_mutex_extension.h
metaprogramming.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
metaprogramming.o: ../threads/rmutex_extension_abstract.h
metaprogramming.o: ../threads/auto_mutex_extension_abstract.h
metaprogramming.o: ../binary_search_tree.h
metaprogramming.o: ../binary_search_tree/binary_search_tree_kernel_1.h
metaprogramming.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
metaprogramming.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
metaprogramming.o: ../interfaces/remover.h ../serialize.h ../algs.h
metaprogramming.o: ../uintn.h ../interfaces/enumerable.h
metaprogramming.o: ../interfaces/map_pair.h ../enable_if.h
metaprogramming.o: ../binary_search_tree/binary_search_tree_kernel_2.h
metaprogramming.o: ../binary_search_tree/binary_search_tree_kernel_c.h
metaprogramming.o: ../assert.h ../../dlib/memory_manager.h
metaprogramming.o: ../member_function_pointer.h
metaprogramming.o: ../member_function_pointer/member_function_pointer_kernel_1.h
metaprogramming.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
metaprogramming.o: ../member_function_pointer/member_function_pointer_kernel_c.h
metaprogramming.o: ../memory_manager.h
metaprogramming.o: ../memory_manager/memory_manager_kernel_1.h
metaprogramming.o: ../memory_manager/memory_manager_kernel_abstract.h
metaprogramming.o: ../memory_manager/memory_manager_kernel_2.h
metaprogramming.o: ../memory_manager/memory_manager_kernel_3.h
metaprogramming.o: ../memory_manager/memory_manager_kernel_2.h
metaprogramming.o: ../binary_search_tree/binary_search_tree_kernel_2.h
metaprogramming.o: ../queue.h ../queue/queue_kernel_1.h
metaprogramming.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
metaprogramming.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
metaprogramming.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
metaprogramming.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
metaprogramming.o: ../set/set_kernel_c.h binary_search_tree.h
metaprogramming.o: ../../dlib/memory_manager_global.h
metaprogramming.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
metaprogramming.o: ../memory_manager/memory_manager_kernel_abstract.h
metaprogramming.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
metaprogramming.o: ../../dlib/memory_manager_stateless.h
metaprogramming.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
metaprogramming.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
metaprogramming.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
metaprogramming.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
metaprogramming.o: ../set/set_compare_abstract.h
metaprogramming.o: ../threads/auto_mutex_extension.h
metaprogramming.o: ../threads/auto_unlock_extension.h
metaprogramming.o: ../threads/auto_unlock_extension_abstract.h
metaprogramming.o: ../threads/create_new_thread_extension.h
metaprogramming.o: ../threads/create_new_thread_extension_abstract.h
metaprogramming.o: ../threads/multithreaded_object_extension.h
metaprogramming.o: ../threads/multithreaded_object_extension_abstract.h
metaprogramming.o: ../threads/rsignaler_extension.h
metaprogramming.o: ../threads/rsignaler_extension_abstract.h ../map.h
metaprogramming.o: ../threads/rmutex_extension.h
metaprogramming.o: ../threads/rsignaler_extension.h
metaprogramming.o: ../threads/threaded_object_extension.h
metaprogramming.o: ../threads/threaded_object_extension_abstract.h
metaprogramming.o: ../threads/thread_specific_data_extension.h
metaprogramming.o: ../threads/thread_specific_data_extension_abstract.h
metaprogramming.o: ../threads/thread_function_extension.h
metaprogramming.o: ../threads/thread_function_extension_abstract.h
metaprogramming.o: ../threads/threaded_object_extension.h ../misc_api.h
metaprogramming.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
metaprogramming.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
metaprogramming.o: ../../dlib/logger/logger_kernel_abstract.h
metaprogramming.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
metaprogramming.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
metaprogramming.o: ../smart_pointers/shared_ptr.h
metaprogramming.o: ../smart_pointers/shared_ptr_abstract.h
metaprogramming.o: ../smart_pointers/weak_ptr.h
metaprogramming.o: ../smart_pointers/shared_ptr.h
metaprogramming.o: ../smart_pointers/weak_ptr_abstract.h
metaprogramming.o: ../../dlib/logger/extra_logger_headers.h
metaprogramming.o: ../../dlib/logger/logger_kernel_1.h
metaprogramming.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
metaprogramming.o: ../config_reader/config_reader_kernel_1.h
metaprogramming.o: ../config_reader/config_reader_kernel_abstract.h
metaprogramming.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
metaprogramming.o: ../../dlib/map/map_kernel_abstract.h
metaprogramming.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
metaprogramming.o: ../tokenizer/tokenizer_kernel_1.h
metaprogramming.o: ../tokenizer/tokenizer_kernel_abstract.h
metaprogramming.o: ../tokenizer/tokenizer_kernel_c.h
metaprogramming.o: ../config_reader/config_reader_thread_safe_1.h
metaprogramming.o: ../config_reader/config_reader_thread_safe_abstract.h
metaprogramming.o: ../../dlib/assert.h
multithreaded_object.o: ../../dlib/threads.h tester.h ../../dlib/map.h
multithreaded_object.o: ../../dlib/logger.h
multithreaded_object.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
multithreaded_object.o: ../threads/threads_kernel.h ../platform.h
multithreaded_object.o: ../threads/posix.h ../threads/threads_kernel_2.h
multithreaded_object.o: ../threads/threads_kernel_abstract.h
multithreaded_object.o: /usr/include/pthread.h /usr/include/features.h
multithreaded_object.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
multithreaded_object.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
multithreaded_object.o: /usr/include/sched.h /usr/include/bits/types.h
multithreaded_object.o: /usr/include/bits/typesizes.h /usr/include/time.h
multithreaded_object.o: /usr/include/bits/sched.h /usr/include/signal.h
multithreaded_object.o: /usr/include/bits/sigset.h
multithreaded_object.o: /usr/include/bits/pthreadtypes.h
multithreaded_object.o: /usr/include/bits/setjmp.h /usr/include/errno.h
multithreaded_object.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
multithreaded_object.o: /usr/include/asm/errno.h
multithreaded_object.o: /usr/include/asm-i386/errno.h
multithreaded_object.o: /usr/include/asm-generic/errno.h
multithreaded_object.o: /usr/include/asm-generic/errno-base.h
multithreaded_object.o: /usr/include/sys/time.h /usr/include/bits/time.h
multithreaded_object.o: /usr/include/sys/select.h /usr/include/bits/select.h
multithreaded_object.o: ../algs.h ../platform.h ../assert.h ../error.h
multithreaded_object.o: ../noncopyable.h ../threads/threads_kernel_shared.h
multithreaded_object.o: ../threads/auto_mutex_extension.h
multithreaded_object.o: ../threads/threads_kernel.h
multithreaded_object.o: ../threads/rmutex_extension.h
multithreaded_object.o: ../threads/rmutex_extension_abstract.h
multithreaded_object.o: ../threads/auto_mutex_extension_abstract.h
multithreaded_object.o: ../binary_search_tree.h
multithreaded_object.o: ../binary_search_tree/binary_search_tree_kernel_1.h
multithreaded_object.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
multithreaded_object.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
multithreaded_object.o: ../interfaces/remover.h ../serialize.h ../algs.h
multithreaded_object.o: ../uintn.h ../interfaces/enumerable.h
multithreaded_object.o: ../interfaces/map_pair.h ../enable_if.h
multithreaded_object.o: ../binary_search_tree/binary_search_tree_kernel_2.h
multithreaded_object.o: ../binary_search_tree/binary_search_tree_kernel_c.h
multithreaded_object.o: ../assert.h ../../dlib/memory_manager.h
multithreaded_object.o: ../member_function_pointer.h
multithreaded_object.o: ../member_function_pointer/member_function_pointer_kernel_1.h
multithreaded_object.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
multithreaded_object.o: ../member_function_pointer/member_function_pointer_kernel_c.h
multithreaded_object.o: ../memory_manager.h
multithreaded_object.o: ../memory_manager/memory_manager_kernel_1.h
multithreaded_object.o: ../memory_manager/memory_manager_kernel_abstract.h
multithreaded_object.o: ../memory_manager/memory_manager_kernel_2.h
multithreaded_object.o: ../memory_manager/memory_manager_kernel_3.h
multithreaded_object.o: ../memory_manager/memory_manager_kernel_2.h
multithreaded_object.o: ../binary_search_tree/binary_search_tree_kernel_2.h
multithreaded_object.o: ../queue.h ../queue/queue_kernel_1.h
multithreaded_object.o: ../queue/queue_kernel_abstract.h
multithreaded_object.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
multithreaded_object.o: ../queue/queue_sort_1.h
multithreaded_object.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
multithreaded_object.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
multithreaded_object.o: ../set/set_kernel_c.h binary_search_tree.h
multithreaded_object.o: ../../dlib/memory_manager_global.h
multithreaded_object.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
multithreaded_object.o: ../memory_manager/memory_manager_kernel_abstract.h
multithreaded_object.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
multithreaded_object.o: ../../dlib/memory_manager_stateless.h
multithreaded_object.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
multithreaded_object.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
multithreaded_object.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
multithreaded_object.o: ../../dlib/binary_search_tree.h
multithreaded_object.o: ../set/set_compare_1.h ../set/set_compare_abstract.h
multithreaded_object.o: ../threads/auto_mutex_extension.h
multithreaded_object.o: ../threads/auto_unlock_extension.h
multithreaded_object.o: ../threads/auto_unlock_extension_abstract.h
multithreaded_object.o: ../threads/create_new_thread_extension.h
multithreaded_object.o: ../threads/create_new_thread_extension_abstract.h
multithreaded_object.o: ../threads/multithreaded_object_extension.h
multithreaded_object.o: ../threads/multithreaded_object_extension_abstract.h
multithreaded_object.o: ../threads/rsignaler_extension.h
multithreaded_object.o: ../threads/rsignaler_extension_abstract.h ../map.h
multithreaded_object.o: ../threads/rmutex_extension.h
multithreaded_object.o: ../threads/rsignaler_extension.h
multithreaded_object.o: ../threads/threaded_object_extension.h
multithreaded_object.o: ../threads/threaded_object_extension_abstract.h
multithreaded_object.o: ../threads/thread_specific_data_extension.h
multithreaded_object.o: ../threads/thread_specific_data_extension_abstract.h
multithreaded_object.o: ../threads/thread_function_extension.h
multithreaded_object.o: ../threads/thread_function_extension_abstract.h
multithreaded_object.o: ../threads/threaded_object_extension.h ../misc_api.h
multithreaded_object.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
multithreaded_object.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
multithreaded_object.o: ../../dlib/logger/logger_kernel_abstract.h
multithreaded_object.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
multithreaded_object.o: ../noncopyable.h
multithreaded_object.o: ../smart_pointers/scoped_ptr_abstract.h
multithreaded_object.o: ../smart_pointers/shared_ptr.h
multithreaded_object.o: ../smart_pointers/shared_ptr_abstract.h
multithreaded_object.o: ../smart_pointers/weak_ptr.h
multithreaded_object.o: ../smart_pointers/shared_ptr.h
multithreaded_object.o: ../smart_pointers/weak_ptr_abstract.h
multithreaded_object.o: ../../dlib/logger/extra_logger_headers.h
multithreaded_object.o: ../../dlib/logger/logger_kernel_1.h
multithreaded_object.o: ../../dlib/logger/logger_config_file.h
multithreaded_object.o: ../config_reader.h
multithreaded_object.o: ../config_reader/config_reader_kernel_1.h
multithreaded_object.o: ../config_reader/config_reader_kernel_abstract.h
multithreaded_object.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
multithreaded_object.o: ../../dlib/map/map_kernel_abstract.h
multithreaded_object.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
multithreaded_object.o: ../tokenizer/tokenizer_kernel_1.h
multithreaded_object.o: ../tokenizer/tokenizer_kernel_abstract.h
multithreaded_object.o: ../tokenizer/tokenizer_kernel_c.h
multithreaded_object.o: ../config_reader/config_reader_thread_safe_1.h
multithreaded_object.o: ../config_reader/config_reader_thread_safe_abstract.h
multithreaded_object.o: ../../dlib/assert.h ../../dlib/algs.h
pipe.o: ../../dlib/misc_api.h ../platform.h ../misc_api/posix.h
pipe.o: ../misc_api/misc_api_kernel_2.h
pipe.o: ../misc_api/misc_api_kernel_abstract.h ../algs.h ../assert.h
pipe.o: ../error.h ../noncopyable.h ../uintn.h ../../dlib/pipe.h
pipe.o: ../../dlib/pipe/pipe_kernel_1.h ../threads.h
pipe.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
pipe.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
pipe.o: /usr/include/pthread.h /usr/include/features.h
pipe.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
pipe.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
pipe.o: /usr/include/sched.h /usr/include/bits/types.h
pipe.o: /usr/include/bits/typesizes.h /usr/include/time.h
pipe.o: /usr/include/bits/sched.h /usr/include/signal.h
pipe.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
pipe.o: /usr/include/bits/setjmp.h /usr/include/errno.h
pipe.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
pipe.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
pipe.o: /usr/include/asm-generic/errno.h
pipe.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
pipe.o: /usr/include/bits/time.h /usr/include/sys/select.h
pipe.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
pipe.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
pipe.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
pipe.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
pipe.o: ../binary_search_tree/binary_search_tree_kernel_1.h
pipe.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
pipe.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
pipe.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
pipe.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
pipe.o: ../binary_search_tree/binary_search_tree_kernel_2.h
pipe.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
pipe.o: ../../dlib/memory_manager.h ../member_function_pointer.h
pipe.o: ../member_function_pointer/member_function_pointer_kernel_1.h
pipe.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
pipe.o: ../member_function_pointer/member_function_pointer_kernel_c.h
pipe.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
pipe.o: ../memory_manager/memory_manager_kernel_abstract.h
pipe.o: ../memory_manager/memory_manager_kernel_2.h
pipe.o: ../memory_manager/memory_manager_kernel_3.h
pipe.o: ../memory_manager/memory_manager_kernel_2.h
pipe.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
pipe.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
pipe.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
pipe.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
pipe.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
pipe.o: ../set/set_kernel_c.h binary_search_tree.h
pipe.o: ../../dlib/memory_manager_global.h
pipe.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
pipe.o: ../memory_manager/memory_manager_kernel_abstract.h
pipe.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
pipe.o: ../../dlib/memory_manager_stateless.h
pipe.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
pipe.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
pipe.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
pipe.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
pipe.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../misc_api.h
pipe.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
pipe.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h ../noncopyable.h
pipe.o: ../smart_pointers/scoped_ptr_abstract.h
pipe.o: ../smart_pointers/shared_ptr.h
pipe.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
pipe.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
pipe.o: ../../dlib/logger/extra_logger_headers.h
pipe.o: ../../dlib/logger/logger_kernel_1.h
pipe.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
pipe.o: ../config_reader/config_reader_kernel_1.h
pipe.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
pipe.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
pipe.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
pipe.o: ../tokenizer/tokenizer_kernel_1.h
pipe.o: ../tokenizer/tokenizer_kernel_abstract.h
pipe.o: ../tokenizer/tokenizer_kernel_c.h
pipe.o: ../config_reader/config_reader_thread_safe_1.h
pipe.o: ../config_reader/config_reader_thread_safe_abstract.h
pipe.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
pipe.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
pipe.o: ../threads/auto_unlock_extension.h
pipe.o: ../threads/auto_unlock_extension_abstract.h
pipe.o: ../threads/create_new_thread_extension.h
pipe.o: ../threads/create_new_thread_extension_abstract.h
pipe.o: ../threads/multithreaded_object_extension.h
pipe.o: ../threads/multithreaded_object_extension_abstract.h
pipe.o: ../threads/rsignaler_extension.h
pipe.o: ../threads/rsignaler_extension_abstract.h
pipe.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
pipe.o: ../threads/threaded_object_extension.h
pipe.o: ../threads/threaded_object_extension_abstract.h
pipe.o: ../threads/thread_specific_data_extension.h
pipe.o: ../threads/thread_specific_data_extension_abstract.h
pipe.o: ../threads/thread_function_extension.h
pipe.o: ../threads/thread_function_extension_abstract.h
pipe.o: ../threads/threaded_object_extension.h
pipe.o: ../../dlib/pipe/pipe_kernel_abstract.h
pixel.o: ../../dlib/pixel.h ../serialize.h ../algs.h ../uintn.h
pixel.o: ../enable_if.h ../../dlib/matrix.h ../matrix/matrix.h
pixel.o: ../matrix/matrix_abstract.h ../algs.h ../platform.h ../assert.h
pixel.o: ../error.h ../noncopyable.h ../serialize.h
pixel.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
pixel.o: ../matrix/matrix_utilities.h ../matrix/matrix_utilities_abstract.h
pixel.o: ../matrix/matrix.h ../pixel.h ../matrix/matrix_math_functions.h
pixel.o: ../matrix/matrix_utilities.h tester.h ../../dlib/map.h
pixel.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../threads.h
pixel.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
pixel.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
pixel.o: /usr/include/pthread.h /usr/include/features.h
pixel.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
pixel.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
pixel.o: /usr/include/sched.h /usr/include/bits/types.h
pixel.o: /usr/include/bits/typesizes.h /usr/include/time.h
pixel.o: /usr/include/bits/sched.h /usr/include/signal.h
pixel.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
pixel.o: /usr/include/bits/setjmp.h /usr/include/errno.h
pixel.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
pixel.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
pixel.o: /usr/include/asm-generic/errno.h
pixel.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
pixel.o: /usr/include/bits/time.h /usr/include/sys/select.h
pixel.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
pixel.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
pixel.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
pixel.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
pixel.o: ../binary_search_tree/binary_search_tree_kernel_1.h
pixel.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
pixel.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
pixel.o: ../interfaces/remover.h
pixel.o: ../binary_search_tree/binary_search_tree_kernel_2.h
pixel.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
pixel.o: ../../dlib/memory_manager.h ../member_function_pointer.h
pixel.o: ../member_function_pointer/member_function_pointer_kernel_1.h
pixel.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
pixel.o: ../member_function_pointer/member_function_pointer_kernel_c.h
pixel.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
pixel.o: ../memory_manager/memory_manager_kernel_abstract.h
pixel.o: ../memory_manager/memory_manager_kernel_2.h
pixel.o: ../memory_manager/memory_manager_kernel_3.h
pixel.o: ../memory_manager/memory_manager_kernel_2.h
pixel.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
pixel.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
pixel.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
pixel.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
pixel.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
pixel.o: ../set/set_kernel_c.h binary_search_tree.h
pixel.o: ../../dlib/memory_manager_global.h
pixel.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
pixel.o: ../memory_manager/memory_manager_kernel_abstract.h
pixel.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
pixel.o: ../../dlib/memory_manager_stateless.h
pixel.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
pixel.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
pixel.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
pixel.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
pixel.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
pixel.o: ../threads/auto_unlock_extension.h
pixel.o: ../threads/auto_unlock_extension_abstract.h
pixel.o: ../threads/create_new_thread_extension.h
pixel.o: ../threads/create_new_thread_extension_abstract.h
pixel.o: ../threads/multithreaded_object_extension.h
pixel.o: ../threads/multithreaded_object_extension_abstract.h
pixel.o: ../threads/rsignaler_extension.h
pixel.o: ../threads/rsignaler_extension_abstract.h ../map.h
pixel.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
pixel.o: ../threads/threaded_object_extension.h
pixel.o: ../threads/threaded_object_extension_abstract.h
pixel.o: ../threads/thread_specific_data_extension.h
pixel.o: ../threads/thread_specific_data_extension_abstract.h
pixel.o: ../threads/thread_function_extension.h
pixel.o: ../threads/thread_function_extension_abstract.h
pixel.o: ../threads/threaded_object_extension.h ../misc_api.h
pixel.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
pixel.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
pixel.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
pixel.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
pixel.o: ../smart_pointers/scoped_ptr_abstract.h
pixel.o: ../smart_pointers/shared_ptr.h
pixel.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
pixel.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
pixel.o: ../../dlib/logger/extra_logger_headers.h
pixel.o: ../../dlib/logger/logger_kernel_1.h
pixel.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
pixel.o: ../config_reader/config_reader_kernel_1.h
pixel.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
pixel.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
pixel.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
pixel.o: ../tokenizer/tokenizer_kernel_1.h
pixel.o: ../tokenizer/tokenizer_kernel_abstract.h
pixel.o: ../tokenizer/tokenizer_kernel_c.h
pixel.o: ../config_reader/config_reader_thread_safe_1.h
pixel.o: ../config_reader/config_reader_thread_safe_abstract.h
pixel.o: ../../dlib/assert.h ../../dlib/algs.h
queue.o: ../../dlib/queue.h ../queue/queue_kernel_1.h
queue.o: ../queue/queue_kernel_abstract.h ../algs.h ../platform.h ../assert.h
queue.o: ../error.h ../noncopyable.h ../interfaces/enumerable.h
queue.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
queue.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
queue.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
queue.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
queue.o: ../memory_manager/memory_manager_kernel_2.h
queue.o: ../memory_manager/memory_manager_kernel_3.h
queue.o: ../memory_manager/memory_manager_kernel_2.h
queue.o: ../binary_search_tree/binary_search_tree_kernel_2.h
queue.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
queue.o: ../interfaces/map_pair.h ../queue/queue_kernel_2.h
queue.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
queue.o: ../queue/queue_sort_abstract.h ../sort.h ../../dlib/memory_manager.h
queue.o: ../../dlib/memory_manager_global.h
queue.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
queue.o: ../memory_manager/memory_manager_kernel_abstract.h
queue.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
queue.o: tester.h ../../dlib/map.h ../../dlib/logger.h
queue.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
queue.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
queue.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
queue.o: /usr/include/pthread.h /usr/include/features.h
queue.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
queue.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
queue.o: /usr/include/sched.h /usr/include/bits/types.h
queue.o: /usr/include/bits/typesizes.h /usr/include/time.h
queue.o: /usr/include/bits/sched.h /usr/include/signal.h
queue.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
queue.o: /usr/include/bits/setjmp.h /usr/include/errno.h
queue.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
queue.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
queue.o: /usr/include/asm-generic/errno.h
queue.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
queue.o: /usr/include/bits/time.h /usr/include/sys/select.h
queue.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
queue.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
queue.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
queue.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
queue.o: ../binary_search_tree/binary_search_tree_kernel_1.h
queue.o: ../binary_search_tree/binary_search_tree_kernel_2.h
queue.o: ../binary_search_tree/binary_search_tree_kernel_c.h
queue.o: ../member_function_pointer.h
queue.o: ../member_function_pointer/member_function_pointer_kernel_1.h
queue.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
queue.o: ../member_function_pointer/member_function_pointer_kernel_c.h
queue.o: ../queue.h ../set.h ../set/set_kernel_1.h
queue.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
queue.o: binary_search_tree.h ../../dlib/memory_manager_stateless.h
queue.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
queue.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
queue.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
queue.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
queue.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
queue.o: ../threads/auto_unlock_extension.h
queue.o: ../threads/auto_unlock_extension_abstract.h
queue.o: ../threads/create_new_thread_extension.h
queue.o: ../threads/create_new_thread_extension_abstract.h
queue.o: ../threads/multithreaded_object_extension.h
queue.o: ../threads/multithreaded_object_extension_abstract.h
queue.o: ../threads/rsignaler_extension.h
queue.o: ../threads/rsignaler_extension_abstract.h ../map.h
queue.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
queue.o: ../threads/threaded_object_extension.h
queue.o: ../threads/threaded_object_extension_abstract.h
queue.o: ../threads/thread_specific_data_extension.h
queue.o: ../threads/thread_specific_data_extension_abstract.h
queue.o: ../threads/thread_function_extension.h
queue.o: ../threads/thread_function_extension_abstract.h
queue.o: ../threads/threaded_object_extension.h ../misc_api.h
queue.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
queue.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
queue.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
queue.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
queue.o: ../smart_pointers/scoped_ptr_abstract.h
queue.o: ../smart_pointers/shared_ptr.h
queue.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
queue.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
queue.o: ../../dlib/logger/extra_logger_headers.h
queue.o: ../../dlib/logger/logger_kernel_1.h
queue.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
queue.o: ../config_reader/config_reader_kernel_1.h
queue.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
queue.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
queue.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
queue.o: ../tokenizer/tokenizer_kernel_1.h
queue.o: ../tokenizer/tokenizer_kernel_abstract.h
queue.o: ../tokenizer/tokenizer_kernel_c.h
queue.o: ../config_reader/config_reader_thread_safe_1.h
queue.o: ../config_reader/config_reader_thread_safe_abstract.h
queue.o: ../../dlib/assert.h ../../dlib/algs.h
rand.o: ../../dlib/rand.h ../rand/rand_kernel_1.h ../algs.h ../platform.h
rand.o: ../assert.h ../error.h ../noncopyable.h
rand.o: ../rand/rand_kernel_abstract.h ../rand/mersenne_twister.h ../uintn.h
rand.o: ../rand/rand_float_1.h ../rand/rand_float_abstract.h ../algs.h
rand.o: ../../dlib/compress_stream.h
rand.o: ../../dlib/compress_stream/compress_stream_kernel_1.h
rand.o: ../../dlib/compress_stream/compress_stream_kernel_abstract.h
rand.o: ../../dlib/compress_stream/compress_stream_kernel_2.h
rand.o: ../../dlib/compress_stream/compress_stream_kernel_3.h ../assert.h
rand.o: conditioning_class.h ../../dlib/conditioning_class.h
rand.o: ../../dlib/conditioning_class/conditioning_class_kernel_1.h
rand.o: ../../dlib/conditioning_class/conditioning_class_kernel_abstract.h
rand.o: ../../dlib/conditioning_class/conditioning_class_kernel_2.h
rand.o: ../../dlib/conditioning_class/conditioning_class_kernel_3.h
rand.o: ../../dlib/conditioning_class/conditioning_class_kernel_4.h
rand.o: ../../dlib/conditioning_class/conditioning_class_kernel_c.h
rand.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
rand.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../threads.h
rand.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
rand.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
rand.o: /usr/include/pthread.h /usr/include/features.h
rand.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
rand.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
rand.o: /usr/include/sched.h /usr/include/bits/types.h
rand.o: /usr/include/bits/typesizes.h /usr/include/time.h
rand.o: /usr/include/bits/sched.h /usr/include/signal.h
rand.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
rand.o: /usr/include/bits/setjmp.h /usr/include/errno.h
rand.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
rand.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
rand.o: /usr/include/asm-generic/errno.h
rand.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
rand.o: /usr/include/bits/time.h /usr/include/sys/select.h
rand.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
rand.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
rand.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
rand.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
rand.o: ../binary_search_tree/binary_search_tree_kernel_1.h
rand.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
rand.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
rand.o: ../interfaces/remover.h ../serialize.h ../uintn.h
rand.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
rand.o: ../binary_search_tree/binary_search_tree_kernel_2.h
rand.o: ../binary_search_tree/binary_search_tree_kernel_c.h
rand.o: ../member_function_pointer.h
rand.o: ../member_function_pointer/member_function_pointer_kernel_1.h
rand.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
rand.o: ../member_function_pointer/member_function_pointer_kernel_c.h
rand.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
rand.o: ../memory_manager/memory_manager_kernel_abstract.h
rand.o: ../memory_manager/memory_manager_kernel_2.h
rand.o: ../memory_manager/memory_manager_kernel_3.h
rand.o: ../memory_manager/memory_manager_kernel_2.h
rand.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
rand.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
rand.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
rand.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
rand.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
rand.o: ../set/set_kernel_c.h binary_search_tree.h
rand.o: ../../dlib/memory_manager_global.h
rand.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
rand.o: ../memory_manager/memory_manager_kernel_abstract.h
rand.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
rand.o: ../../dlib/memory_manager_stateless.h
rand.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
rand.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
rand.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
rand.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
rand.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
rand.o: ../threads/auto_unlock_extension.h
rand.o: ../threads/auto_unlock_extension_abstract.h
rand.o: ../threads/create_new_thread_extension.h
rand.o: ../threads/create_new_thread_extension_abstract.h
rand.o: ../threads/multithreaded_object_extension.h
rand.o: ../threads/multithreaded_object_extension_abstract.h
rand.o: ../threads/rsignaler_extension.h
rand.o: ../threads/rsignaler_extension_abstract.h ../map.h
rand.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
rand.o: ../threads/threaded_object_extension.h
rand.o: ../threads/threaded_object_extension_abstract.h
rand.o: ../threads/thread_specific_data_extension.h
rand.o: ../threads/thread_specific_data_extension_abstract.h
rand.o: ../threads/thread_function_extension.h
rand.o: ../threads/thread_function_extension_abstract.h
rand.o: ../threads/threaded_object_extension.h ../misc_api.h
rand.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
rand.o: ../misc_api/misc_api_kernel_abstract.h
rand.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
rand.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
rand.o: ../smart_pointers/scoped_ptr_abstract.h
rand.o: ../smart_pointers/shared_ptr.h
rand.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
rand.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
rand.o: ../../dlib/logger/extra_logger_headers.h
rand.o: ../../dlib/logger/logger_kernel_1.h
rand.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
rand.o: ../config_reader/config_reader_kernel_1.h
rand.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
rand.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
rand.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
rand.o: ../tokenizer/tokenizer_kernel_1.h
rand.o: ../tokenizer/tokenizer_kernel_abstract.h
rand.o: ../tokenizer/tokenizer_kernel_c.h
rand.o: ../config_reader/config_reader_thread_safe_1.h
rand.o: ../config_reader/config_reader_thread_safe_abstract.h
rand.o: ../../dlib/assert.h ../../dlib/algs.h ../../dlib/entropy_encoder.h
rand.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_1.h
rand.o: ../entropy_encoder/entropy_encoder_kernel_abstract.h
rand.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_2.h
rand.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_c.h
rand.o: ../../dlib/entropy_decoder.h
rand.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_1.h
rand.o: ../entropy_decoder/entropy_decoder_kernel_abstract.h
rand.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_2.h
rand.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_c.h
rand.o: ../../dlib/entropy_encoder_model.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_1.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_abstract.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_2.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_3.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_4.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_5.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_6.h
rand.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_c.h
rand.o: ../../dlib/sliding_buffer.h
rand.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_1.h
rand.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_abstract.h
rand.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_c.h
rand.o: ../../dlib/entropy_decoder_model.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_1.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_abstract.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_2.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_3.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_4.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_5.h
rand.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_6.h
rand.o: ../../dlib/lz77_buffer.h
rand.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_1.h
rand.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_abstract.h
rand.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_2.h
rand.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_c.h ../../dlib/lzp_buffer.h
rand.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_1.h
rand.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_abstract.h
rand.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_2.h
rand.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_c.h ../../dlib/crc32.h
rand.o: ../../dlib/crc32/crc32_kernel_1.h
rand.o: ../../dlib/crc32/crc32_kernel_abstract.h
reference_counter.o: ../../dlib/reference_counter.h
reference_counter.o: ../../dlib/reference_counter/reference_counter_kernel_1.h
reference_counter.o: ../../dlib/reference_counter/reference_counter_kernel_abstract.h
reference_counter.o: ../algs.h ../platform.h ../assert.h ../error.h
reference_counter.o: ../noncopyable.h ../algs.h tester.h ../../dlib/map.h
reference_counter.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
reference_counter.o: ../threads.h ../threads/threads_kernel.h ../platform.h
reference_counter.o: ../threads/posix.h ../threads/threads_kernel_2.h
reference_counter.o: ../threads/threads_kernel_abstract.h
reference_counter.o: /usr/include/pthread.h /usr/include/features.h
reference_counter.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
reference_counter.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
reference_counter.o: /usr/include/sched.h /usr/include/bits/types.h
reference_counter.o: /usr/include/bits/typesizes.h /usr/include/time.h
reference_counter.o: /usr/include/bits/sched.h /usr/include/signal.h
reference_counter.o: /usr/include/bits/sigset.h
reference_counter.o: /usr/include/bits/pthreadtypes.h
reference_counter.o: /usr/include/bits/setjmp.h /usr/include/errno.h
reference_counter.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
reference_counter.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
reference_counter.o: /usr/include/asm-generic/errno.h
reference_counter.o: /usr/include/asm-generic/errno-base.h
reference_counter.o: /usr/include/sys/time.h /usr/include/bits/time.h
reference_counter.o: /usr/include/sys/select.h /usr/include/bits/select.h
reference_counter.o: ../threads/threads_kernel_shared.h
reference_counter.o: ../threads/auto_mutex_extension.h
reference_counter.o: ../threads/threads_kernel.h
reference_counter.o: ../threads/rmutex_extension.h
reference_counter.o: ../threads/rmutex_extension_abstract.h
reference_counter.o: ../threads/auto_mutex_extension_abstract.h
reference_counter.o: ../binary_search_tree.h
reference_counter.o: ../binary_search_tree/binary_search_tree_kernel_1.h
reference_counter.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
reference_counter.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
reference_counter.o: ../interfaces/remover.h ../serialize.h ../uintn.h
reference_counter.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
reference_counter.o: ../enable_if.h
reference_counter.o: ../binary_search_tree/binary_search_tree_kernel_2.h
reference_counter.o: ../binary_search_tree/binary_search_tree_kernel_c.h
reference_counter.o: ../assert.h ../../dlib/memory_manager.h
reference_counter.o: ../member_function_pointer.h
reference_counter.o: ../member_function_pointer/member_function_pointer_kernel_1.h
reference_counter.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
reference_counter.o: ../member_function_pointer/member_function_pointer_kernel_c.h
reference_counter.o: ../memory_manager.h
reference_counter.o: ../memory_manager/memory_manager_kernel_1.h
reference_counter.o: ../memory_manager/memory_manager_kernel_abstract.h
reference_counter.o: ../memory_manager/memory_manager_kernel_2.h
reference_counter.o: ../memory_manager/memory_manager_kernel_3.h
reference_counter.o: ../memory_manager/memory_manager_kernel_2.h
reference_counter.o: ../binary_search_tree/binary_search_tree_kernel_2.h
reference_counter.o: ../queue.h ../queue/queue_kernel_1.h
reference_counter.o: ../queue/queue_kernel_abstract.h
reference_counter.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
reference_counter.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h
reference_counter.o: ../sort.h ../set.h ../set/set_kernel_1.h
reference_counter.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
reference_counter.o: binary_search_tree.h ../../dlib/memory_manager_global.h
reference_counter.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
reference_counter.o: ../memory_manager/memory_manager_kernel_abstract.h
reference_counter.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
reference_counter.o: ../../dlib/memory_manager_stateless.h
reference_counter.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
reference_counter.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
reference_counter.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
reference_counter.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
reference_counter.o: ../set/set_compare_abstract.h
reference_counter.o: ../threads/auto_mutex_extension.h
reference_counter.o: ../threads/auto_unlock_extension.h
reference_counter.o: ../threads/auto_unlock_extension_abstract.h
reference_counter.o: ../threads/create_new_thread_extension.h
reference_counter.o: ../threads/create_new_thread_extension_abstract.h
reference_counter.o: ../threads/multithreaded_object_extension.h
reference_counter.o: ../threads/multithreaded_object_extension_abstract.h
reference_counter.o: ../threads/rsignaler_extension.h
reference_counter.o: ../threads/rsignaler_extension_abstract.h ../map.h
reference_counter.o: ../threads/rmutex_extension.h
reference_counter.o: ../threads/rsignaler_extension.h
reference_counter.o: ../threads/threaded_object_extension.h
reference_counter.o: ../threads/threaded_object_extension_abstract.h
reference_counter.o: ../threads/thread_specific_data_extension.h
reference_counter.o: ../threads/thread_specific_data_extension_abstract.h
reference_counter.o: ../threads/thread_function_extension.h
reference_counter.o: ../threads/thread_function_extension_abstract.h
reference_counter.o: ../threads/threaded_object_extension.h ../misc_api.h
reference_counter.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
reference_counter.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
reference_counter.o: ../../dlib/logger/logger_kernel_abstract.h
reference_counter.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
reference_counter.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
reference_counter.o: ../smart_pointers/shared_ptr.h
reference_counter.o: ../smart_pointers/shared_ptr_abstract.h
reference_counter.o: ../smart_pointers/weak_ptr.h
reference_counter.o: ../smart_pointers/shared_ptr.h
reference_counter.o: ../smart_pointers/weak_ptr_abstract.h
reference_counter.o: ../../dlib/logger/extra_logger_headers.h
reference_counter.o: ../../dlib/logger/logger_kernel_1.h
reference_counter.o: ../../dlib/logger/logger_config_file.h
reference_counter.o: ../config_reader.h
reference_counter.o: ../config_reader/config_reader_kernel_1.h
reference_counter.o: ../config_reader/config_reader_kernel_abstract.h
reference_counter.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
reference_counter.o: ../../dlib/map/map_kernel_abstract.h
reference_counter.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
reference_counter.o: ../tokenizer/tokenizer_kernel_1.h
reference_counter.o: ../tokenizer/tokenizer_kernel_abstract.h
reference_counter.o: ../tokenizer/tokenizer_kernel_c.h
reference_counter.o: ../config_reader/config_reader_thread_safe_1.h
reference_counter.o: ../config_reader/config_reader_thread_safe_abstract.h
reference_counter.o: ../../dlib/assert.h ../../dlib/algs.h
sequence.o: ../../dlib/sequence.h ../../dlib/sequence/sequence_kernel_1.h
sequence.o: ../../dlib/sequence/sequence_kernel_abstract.h ../algs.h
sequence.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
sequence.o: ../interfaces/enumerable.h ../interfaces/remover.h ../serialize.h
sequence.o: ../algs.h ../uintn.h ../interfaces/enumerable.h
sequence.o: ../interfaces/map_pair.h ../enable_if.h ../memory_manager.h
sequence.o: ../memory_manager/memory_manager_kernel_1.h
sequence.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
sequence.o: ../memory_manager/memory_manager_kernel_2.h
sequence.o: ../memory_manager/memory_manager_kernel_3.h
sequence.o: ../memory_manager/memory_manager_kernel_2.h
sequence.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sequence.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
sequence.o: ../interfaces/map_pair.h ../../dlib/sequence/sequence_kernel_2.h
sequence.o: ../../dlib/sequence/sequence_kernel_c.h
sequence.o: ../../dlib/sequence/sequence_compare_1.h
sequence.o: ../../dlib/sequence/sequence_compare_abstract.h
sequence.o: ../../dlib/sequence/sequence_sort_1.h
sequence.o: ../../dlib/sequence/sequence_sort_abstract.h
sequence.o: ../../dlib/sequence/sequence_sort_2.h ../sort.h
sequence.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
sequence.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
sequence.o: ../threads.h ../threads/threads_kernel.h ../platform.h
sequence.o: ../threads/posix.h ../threads/threads_kernel_2.h
sequence.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
sequence.o: /usr/include/features.h /usr/include/sys/cdefs.h
sequence.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
sequence.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
sequence.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
sequence.o: /usr/include/time.h /usr/include/bits/sched.h
sequence.o: /usr/include/signal.h /usr/include/bits/sigset.h
sequence.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
sequence.o: /usr/include/errno.h /usr/include/bits/errno.h
sequence.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
sequence.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
sequence.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
sequence.o: /usr/include/bits/time.h /usr/include/sys/select.h
sequence.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
sequence.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
sequence.o: ../threads/rmutex_extension.h
sequence.o: ../threads/rmutex_extension_abstract.h
sequence.o: ../threads/auto_mutex_extension_abstract.h
sequence.o: ../binary_search_tree.h
sequence.o: ../binary_search_tree/binary_search_tree_kernel_1.h
sequence.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sequence.o: ../binary_search_tree/binary_search_tree_kernel_c.h
sequence.o: ../member_function_pointer.h
sequence.o: ../member_function_pointer/member_function_pointer_kernel_1.h
sequence.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
sequence.o: ../member_function_pointer/member_function_pointer_kernel_c.h
sequence.o: ../queue.h ../queue/queue_kernel_1.h
sequence.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
sequence.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
sequence.o: ../queue/queue_sort_abstract.h ../set.h ../set/set_kernel_1.h
sequence.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
sequence.o: binary_search_tree.h ../../dlib/memory_manager_global.h
sequence.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
sequence.o: ../memory_manager/memory_manager_kernel_abstract.h
sequence.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
sequence.o: ../../dlib/memory_manager_stateless.h
sequence.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
sequence.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
sequence.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
sequence.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
sequence.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
sequence.o: ../threads/auto_unlock_extension.h
sequence.o: ../threads/auto_unlock_extension_abstract.h
sequence.o: ../threads/create_new_thread_extension.h
sequence.o: ../threads/create_new_thread_extension_abstract.h
sequence.o: ../threads/multithreaded_object_extension.h
sequence.o: ../threads/multithreaded_object_extension_abstract.h
sequence.o: ../threads/rsignaler_extension.h
sequence.o: ../threads/rsignaler_extension_abstract.h ../map.h
sequence.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
sequence.o: ../threads/threaded_object_extension.h
sequence.o: ../threads/threaded_object_extension_abstract.h
sequence.o: ../threads/thread_specific_data_extension.h
sequence.o: ../threads/thread_specific_data_extension_abstract.h
sequence.o: ../threads/thread_function_extension.h
sequence.o: ../threads/thread_function_extension_abstract.h
sequence.o: ../threads/threaded_object_extension.h ../misc_api.h
sequence.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
sequence.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
sequence.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
sequence.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
sequence.o: ../smart_pointers/scoped_ptr_abstract.h
sequence.o: ../smart_pointers/shared_ptr.h
sequence.o: ../smart_pointers/shared_ptr_abstract.h
sequence.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
sequence.o: ../smart_pointers/weak_ptr_abstract.h
sequence.o: ../../dlib/logger/extra_logger_headers.h
sequence.o: ../../dlib/logger/logger_kernel_1.h
sequence.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
sequence.o: ../config_reader/config_reader_kernel_1.h
sequence.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
sequence.o: ../../dlib/map/map_kernel_1.h
sequence.o: ../../dlib/map/map_kernel_abstract.h
sequence.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
sequence.o: ../tokenizer/tokenizer_kernel_1.h
sequence.o: ../tokenizer/tokenizer_kernel_abstract.h
sequence.o: ../tokenizer/tokenizer_kernel_c.h
sequence.o: ../config_reader/config_reader_thread_safe_1.h
sequence.o: ../config_reader/config_reader_thread_safe_abstract.h
sequence.o: ../../dlib/assert.h ../../dlib/algs.h
serialize.o: ../../dlib/compress_stream.h
serialize.o: ../../dlib/compress_stream/compress_stream_kernel_1.h ../algs.h
serialize.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
serialize.o: ../../dlib/compress_stream/compress_stream_kernel_abstract.h
serialize.o: ../../dlib/compress_stream/compress_stream_kernel_2.h
serialize.o: ../../dlib/compress_stream/compress_stream_kernel_3.h
serialize.o: ../assert.h conditioning_class.h ../../dlib/conditioning_class.h
serialize.o: ../../dlib/conditioning_class/conditioning_class_kernel_1.h
serialize.o: ../../dlib/conditioning_class/conditioning_class_kernel_abstract.h
serialize.o: ../../dlib/conditioning_class/conditioning_class_kernel_2.h
serialize.o: ../../dlib/conditioning_class/conditioning_class_kernel_3.h
serialize.o: ../../dlib/conditioning_class/conditioning_class_kernel_4.h
serialize.o: ../../dlib/conditioning_class/conditioning_class_kernel_c.h
serialize.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
serialize.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
serialize.o: ../threads.h ../threads/threads_kernel.h ../platform.h
serialize.o: ../threads/posix.h ../threads/threads_kernel_2.h
serialize.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
serialize.o: /usr/include/features.h /usr/include/sys/cdefs.h
serialize.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
serialize.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
serialize.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
serialize.o: /usr/include/time.h /usr/include/bits/sched.h
serialize.o: /usr/include/signal.h /usr/include/bits/sigset.h
serialize.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
serialize.o: /usr/include/errno.h /usr/include/bits/errno.h
serialize.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
serialize.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
serialize.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
serialize.o: /usr/include/bits/time.h /usr/include/sys/select.h
serialize.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
serialize.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
serialize.o: ../threads/rmutex_extension.h
serialize.o: ../threads/rmutex_extension_abstract.h
serialize.o: ../threads/auto_mutex_extension_abstract.h
serialize.o: ../binary_search_tree.h
serialize.o: ../binary_search_tree/binary_search_tree_kernel_1.h
serialize.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
serialize.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
serialize.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
serialize.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
serialize.o: ../enable_if.h
serialize.o: ../binary_search_tree/binary_search_tree_kernel_2.h
serialize.o: ../binary_search_tree/binary_search_tree_kernel_c.h
serialize.o: ../member_function_pointer.h
serialize.o: ../member_function_pointer/member_function_pointer_kernel_1.h
serialize.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
serialize.o: ../member_function_pointer/member_function_pointer_kernel_c.h
serialize.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
serialize.o: ../memory_manager/memory_manager_kernel_abstract.h
serialize.o: ../memory_manager/memory_manager_kernel_2.h
serialize.o: ../memory_manager/memory_manager_kernel_3.h
serialize.o: ../memory_manager/memory_manager_kernel_2.h
serialize.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
serialize.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
serialize.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
serialize.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
serialize.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
serialize.o: ../set/set_kernel_c.h binary_search_tree.h
serialize.o: ../../dlib/memory_manager_global.h
serialize.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
serialize.o: ../memory_manager/memory_manager_kernel_abstract.h
serialize.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
serialize.o: ../../dlib/memory_manager_stateless.h
serialize.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
serialize.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
serialize.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
serialize.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
serialize.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
serialize.o: ../threads/auto_unlock_extension.h
serialize.o: ../threads/auto_unlock_extension_abstract.h
serialize.o: ../threads/create_new_thread_extension.h
serialize.o: ../threads/create_new_thread_extension_abstract.h
serialize.o: ../threads/multithreaded_object_extension.h
serialize.o: ../threads/multithreaded_object_extension_abstract.h
serialize.o: ../threads/rsignaler_extension.h
serialize.o: ../threads/rsignaler_extension_abstract.h ../map.h
serialize.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
serialize.o: ../threads/threaded_object_extension.h
serialize.o: ../threads/threaded_object_extension_abstract.h
serialize.o: ../threads/thread_specific_data_extension.h
serialize.o: ../threads/thread_specific_data_extension_abstract.h
serialize.o: ../threads/thread_function_extension.h
serialize.o: ../threads/thread_function_extension_abstract.h
serialize.o: ../threads/threaded_object_extension.h ../misc_api.h
serialize.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
serialize.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
serialize.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
serialize.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
serialize.o: ../smart_pointers/scoped_ptr_abstract.h
serialize.o: ../smart_pointers/shared_ptr.h
serialize.o: ../smart_pointers/shared_ptr_abstract.h
serialize.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
serialize.o: ../smart_pointers/weak_ptr_abstract.h
serialize.o: ../../dlib/logger/extra_logger_headers.h
serialize.o: ../../dlib/logger/logger_kernel_1.h
serialize.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
serialize.o: ../config_reader/config_reader_kernel_1.h
serialize.o: ../config_reader/config_reader_kernel_abstract.h
serialize.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
serialize.o: ../../dlib/map/map_kernel_abstract.h
serialize.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
serialize.o: ../tokenizer/tokenizer_kernel_1.h
serialize.o: ../tokenizer/tokenizer_kernel_abstract.h
serialize.o: ../tokenizer/tokenizer_kernel_c.h
serialize.o: ../config_reader/config_reader_thread_safe_1.h
serialize.o: ../config_reader/config_reader_thread_safe_abstract.h
serialize.o: ../../dlib/assert.h ../../dlib/algs.h
serialize.o: ../../dlib/entropy_encoder.h
serialize.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_1.h
serialize.o: ../entropy_encoder/entropy_encoder_kernel_abstract.h
serialize.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_2.h
serialize.o: ../../dlib/entropy_encoder/entropy_encoder_kernel_c.h
serialize.o: ../../dlib/entropy_decoder.h
serialize.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_1.h
serialize.o: ../entropy_decoder/entropy_decoder_kernel_abstract.h
serialize.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_2.h
serialize.o: ../../dlib/entropy_decoder/entropy_decoder_kernel_c.h
serialize.o: ../../dlib/entropy_encoder_model.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_1.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_abstract.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_2.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_3.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_4.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_5.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_6.h
serialize.o: ../../dlib/entropy_encoder_model/entropy_encoder_model_kernel_c.h
serialize.o: ../../dlib/sliding_buffer.h
serialize.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_1.h
serialize.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_abstract.h
serialize.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_c.h
serialize.o: ../../dlib/entropy_decoder_model.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_1.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_abstract.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_2.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_3.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_4.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_5.h
serialize.o: ../../dlib/entropy_decoder_model/entropy_decoder_model_kernel_6.h
serialize.o: ../../dlib/lz77_buffer.h
serialize.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_1.h
serialize.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_abstract.h
serialize.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_2.h
serialize.o: ../../dlib/lz77_buffer/lz77_buffer_kernel_c.h
serialize.o: ../../dlib/lzp_buffer.h
serialize.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_1.h
serialize.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_abstract.h
serialize.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_2.h
serialize.o: ../../dlib/lzp_buffer/lzp_buffer_kernel_c.h ../../dlib/crc32.h
serialize.o: ../../dlib/crc32/crc32_kernel_1.h
serialize.o: ../../dlib/crc32/crc32_kernel_abstract.h ../../dlib/base64.h
serialize.o: ../../dlib/base64/base64_kernel_1.h ../../dlib/serialize.h
set.o: ../../dlib/set.h tester.h ../../dlib/map.h ../../dlib/logger.h
set.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
set.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
set.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
set.o: /usr/include/pthread.h /usr/include/features.h
set.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
set.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
set.o: /usr/include/sched.h /usr/include/bits/types.h
set.o: /usr/include/bits/typesizes.h /usr/include/time.h
set.o: /usr/include/bits/sched.h /usr/include/signal.h
set.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
set.o: /usr/include/bits/setjmp.h /usr/include/errno.h
set.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
set.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
set.o: /usr/include/asm-generic/errno.h /usr/include/asm-generic/errno-base.h
set.o: /usr/include/sys/time.h /usr/include/bits/time.h
set.o: /usr/include/sys/select.h /usr/include/bits/select.h ../algs.h
set.o: ../platform.h ../assert.h ../error.h ../noncopyable.h
set.o: ../threads/threads_kernel_shared.h ../threads/auto_mutex_extension.h
set.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
set.o: ../threads/rmutex_extension_abstract.h
set.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
set.o: ../binary_search_tree/binary_search_tree_kernel_1.h
set.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
set.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
set.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
set.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
set.o: ../binary_search_tree/binary_search_tree_kernel_2.h
set.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
set.o: ../../dlib/memory_manager.h ../member_function_pointer.h
set.o: ../member_function_pointer/member_function_pointer_kernel_1.h
set.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
set.o: ../member_function_pointer/member_function_pointer_kernel_c.h
set.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
set.o: ../memory_manager/memory_manager_kernel_abstract.h
set.o: ../memory_manager/memory_manager_kernel_2.h
set.o: ../memory_manager/memory_manager_kernel_3.h
set.o: ../memory_manager/memory_manager_kernel_2.h
set.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
set.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
set.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
set.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
set.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
set.o: ../set/set_kernel_c.h binary_search_tree.h
set.o: ../../dlib/memory_manager_global.h
set.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
set.o: ../memory_manager/memory_manager_kernel_abstract.h
set.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
set.o: ../../dlib/memory_manager_stateless.h
set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
set.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
set.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
set.o: ../threads/auto_unlock_extension.h
set.o: ../threads/auto_unlock_extension_abstract.h
set.o: ../threads/create_new_thread_extension.h
set.o: ../threads/create_new_thread_extension_abstract.h
set.o: ../threads/multithreaded_object_extension.h
set.o: ../threads/multithreaded_object_extension_abstract.h
set.o: ../threads/rsignaler_extension.h
set.o: ../threads/rsignaler_extension_abstract.h ../map.h
set.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
set.o: ../threads/threaded_object_extension.h
set.o: ../threads/threaded_object_extension_abstract.h
set.o: ../threads/thread_specific_data_extension.h
set.o: ../threads/thread_specific_data_extension_abstract.h
set.o: ../threads/thread_function_extension.h
set.o: ../threads/thread_function_extension_abstract.h
set.o: ../threads/threaded_object_extension.h ../misc_api.h
set.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
set.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
set.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
set.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
set.o: ../smart_pointers/scoped_ptr_abstract.h ../smart_pointers/shared_ptr.h
set.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
set.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
set.o: ../../dlib/logger/extra_logger_headers.h
set.o: ../../dlib/logger/logger_kernel_1.h
set.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
set.o: ../config_reader/config_reader_kernel_1.h
set.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
set.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
set.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
set.o: ../tokenizer/tokenizer_kernel_1.h
set.o: ../tokenizer/tokenizer_kernel_abstract.h
set.o: ../tokenizer/tokenizer_kernel_c.h
set.o: ../config_reader/config_reader_thread_safe_1.h
set.o: ../config_reader/config_reader_thread_safe_abstract.h
set.o: ../../dlib/assert.h ../../dlib/algs.h
sliding_buffer.o: ../../dlib/sliding_buffer.h
sliding_buffer.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_1.h
sliding_buffer.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_abstract.h
sliding_buffer.o: ../algs.h ../platform.h ../assert.h ../error.h
sliding_buffer.o: ../noncopyable.h ../interfaces/enumerable.h ../serialize.h
sliding_buffer.o: ../algs.h ../uintn.h ../interfaces/enumerable.h
sliding_buffer.o: ../interfaces/map_pair.h ../enable_if.h
sliding_buffer.o: ../../dlib/sliding_buffer/sliding_buffer_kernel_c.h
sliding_buffer.o: ../assert.h tester.h ../../dlib/map.h ../../dlib/logger.h
sliding_buffer.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
sliding_buffer.o: ../threads/threads_kernel.h ../platform.h
sliding_buffer.o: ../threads/posix.h ../threads/threads_kernel_2.h
sliding_buffer.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
sliding_buffer.o: /usr/include/features.h /usr/include/sys/cdefs.h
sliding_buffer.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
sliding_buffer.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
sliding_buffer.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
sliding_buffer.o: /usr/include/time.h /usr/include/bits/sched.h
sliding_buffer.o: /usr/include/signal.h /usr/include/bits/sigset.h
sliding_buffer.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
sliding_buffer.o: /usr/include/errno.h /usr/include/bits/errno.h
sliding_buffer.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
sliding_buffer.o: /usr/include/asm-i386/errno.h
sliding_buffer.o: /usr/include/asm-generic/errno.h
sliding_buffer.o: /usr/include/asm-generic/errno-base.h
sliding_buffer.o: /usr/include/sys/time.h /usr/include/bits/time.h
sliding_buffer.o: /usr/include/sys/select.h /usr/include/bits/select.h
sliding_buffer.o: ../threads/threads_kernel_shared.h
sliding_buffer.o: ../threads/auto_mutex_extension.h
sliding_buffer.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
sliding_buffer.o: ../threads/rmutex_extension_abstract.h
sliding_buffer.o: ../threads/auto_mutex_extension_abstract.h
sliding_buffer.o: ../binary_search_tree.h
sliding_buffer.o: ../binary_search_tree/binary_search_tree_kernel_1.h
sliding_buffer.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
sliding_buffer.o: ../interfaces/map_pair.h ../interfaces/remover.h
sliding_buffer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sliding_buffer.o: ../binary_search_tree/binary_search_tree_kernel_c.h
sliding_buffer.o: ../../dlib/memory_manager.h ../member_function_pointer.h
sliding_buffer.o: ../member_function_pointer/member_function_pointer_kernel_1.h
sliding_buffer.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
sliding_buffer.o: ../member_function_pointer/member_function_pointer_kernel_c.h
sliding_buffer.o: ../memory_manager.h
sliding_buffer.o: ../memory_manager/memory_manager_kernel_1.h
sliding_buffer.o: ../memory_manager/memory_manager_kernel_abstract.h
sliding_buffer.o: ../memory_manager/memory_manager_kernel_2.h
sliding_buffer.o: ../memory_manager/memory_manager_kernel_3.h
sliding_buffer.o: ../memory_manager/memory_manager_kernel_2.h
sliding_buffer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sliding_buffer.o: ../queue.h ../queue/queue_kernel_1.h
sliding_buffer.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
sliding_buffer.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
sliding_buffer.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
sliding_buffer.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
sliding_buffer.o: ../set/set_kernel_c.h binary_search_tree.h
sliding_buffer.o: ../../dlib/memory_manager_global.h
sliding_buffer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
sliding_buffer.o: ../memory_manager/memory_manager_kernel_abstract.h
sliding_buffer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
sliding_buffer.o: ../../dlib/memory_manager_stateless.h
sliding_buffer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
sliding_buffer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
sliding_buffer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
sliding_buffer.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
sliding_buffer.o: ../set/set_compare_abstract.h
sliding_buffer.o: ../threads/auto_mutex_extension.h
sliding_buffer.o: ../threads/auto_unlock_extension.h
sliding_buffer.o: ../threads/auto_unlock_extension_abstract.h
sliding_buffer.o: ../threads/create_new_thread_extension.h
sliding_buffer.o: ../threads/create_new_thread_extension_abstract.h
sliding_buffer.o: ../threads/multithreaded_object_extension.h
sliding_buffer.o: ../threads/multithreaded_object_extension_abstract.h
sliding_buffer.o: ../threads/rsignaler_extension.h
sliding_buffer.o: ../threads/rsignaler_extension_abstract.h ../map.h
sliding_buffer.o: ../threads/rmutex_extension.h
sliding_buffer.o: ../threads/rsignaler_extension.h
sliding_buffer.o: ../threads/threaded_object_extension.h
sliding_buffer.o: ../threads/threaded_object_extension_abstract.h
sliding_buffer.o: ../threads/thread_specific_data_extension.h
sliding_buffer.o: ../threads/thread_specific_data_extension_abstract.h
sliding_buffer.o: ../threads/thread_function_extension.h
sliding_buffer.o: ../threads/thread_function_extension_abstract.h
sliding_buffer.o: ../threads/threaded_object_extension.h ../misc_api.h
sliding_buffer.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
sliding_buffer.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
sliding_buffer.o: ../../dlib/logger/logger_kernel_abstract.h
sliding_buffer.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
sliding_buffer.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
sliding_buffer.o: ../smart_pointers/shared_ptr.h
sliding_buffer.o: ../smart_pointers/shared_ptr_abstract.h
sliding_buffer.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
sliding_buffer.o: ../smart_pointers/weak_ptr_abstract.h
sliding_buffer.o: ../../dlib/logger/extra_logger_headers.h
sliding_buffer.o: ../../dlib/logger/logger_kernel_1.h
sliding_buffer.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
sliding_buffer.o: ../config_reader/config_reader_kernel_1.h
sliding_buffer.o: ../config_reader/config_reader_kernel_abstract.h
sliding_buffer.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
sliding_buffer.o: ../../dlib/map/map_kernel_abstract.h
sliding_buffer.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
sliding_buffer.o: ../tokenizer/tokenizer_kernel_1.h
sliding_buffer.o: ../tokenizer/tokenizer_kernel_abstract.h
sliding_buffer.o: ../tokenizer/tokenizer_kernel_c.h
sliding_buffer.o: ../config_reader/config_reader_thread_safe_1.h
sliding_buffer.o: ../config_reader/config_reader_thread_safe_abstract.h
sliding_buffer.o: ../../dlib/assert.h ../../dlib/algs.h
smart_pointers.o: ../../dlib/smart_pointers.h ../smart_pointers/scoped_ptr.h
smart_pointers.o: ../noncopyable.h ../algs.h ../platform.h ../assert.h
smart_pointers.o: ../error.h ../noncopyable.h
smart_pointers.o: ../smart_pointers/scoped_ptr_abstract.h
smart_pointers.o: ../smart_pointers/shared_ptr.h
smart_pointers.o: ../smart_pointers/shared_ptr_abstract.h
smart_pointers.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
smart_pointers.o: ../smart_pointers/weak_ptr_abstract.h tester.h
smart_pointers.o: ../../dlib/map.h ../../dlib/logger.h
smart_pointers.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
smart_pointers.o: ../threads/threads_kernel.h ../platform.h
smart_pointers.o: ../threads/posix.h ../threads/threads_kernel_2.h
smart_pointers.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
smart_pointers.o: /usr/include/features.h /usr/include/sys/cdefs.h
smart_pointers.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
smart_pointers.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
smart_pointers.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
smart_pointers.o: /usr/include/time.h /usr/include/bits/sched.h
smart_pointers.o: /usr/include/signal.h /usr/include/bits/sigset.h
smart_pointers.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
smart_pointers.o: /usr/include/errno.h /usr/include/bits/errno.h
smart_pointers.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
smart_pointers.o: /usr/include/asm-i386/errno.h
smart_pointers.o: /usr/include/asm-generic/errno.h
smart_pointers.o: /usr/include/asm-generic/errno-base.h
smart_pointers.o: /usr/include/sys/time.h /usr/include/bits/time.h
smart_pointers.o: /usr/include/sys/select.h /usr/include/bits/select.h
smart_pointers.o: ../threads/threads_kernel_shared.h
smart_pointers.o: ../threads/auto_mutex_extension.h
smart_pointers.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
smart_pointers.o: ../threads/rmutex_extension_abstract.h
smart_pointers.o: ../threads/auto_mutex_extension_abstract.h
smart_pointers.o: ../binary_search_tree.h
smart_pointers.o: ../binary_search_tree/binary_search_tree_kernel_1.h
smart_pointers.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
smart_pointers.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
smart_pointers.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
smart_pointers.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
smart_pointers.o: ../enable_if.h
smart_pointers.o: ../binary_search_tree/binary_search_tree_kernel_2.h
smart_pointers.o: ../binary_search_tree/binary_search_tree_kernel_c.h
smart_pointers.o: ../assert.h ../../dlib/memory_manager.h
smart_pointers.o: ../member_function_pointer.h
smart_pointers.o: ../member_function_pointer/member_function_pointer_kernel_1.h
smart_pointers.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
smart_pointers.o: ../member_function_pointer/member_function_pointer_kernel_c.h
smart_pointers.o: ../memory_manager.h
smart_pointers.o: ../memory_manager/memory_manager_kernel_1.h
smart_pointers.o: ../memory_manager/memory_manager_kernel_abstract.h
smart_pointers.o: ../memory_manager/memory_manager_kernel_2.h
smart_pointers.o: ../memory_manager/memory_manager_kernel_3.h
smart_pointers.o: ../memory_manager/memory_manager_kernel_2.h
smart_pointers.o: ../binary_search_tree/binary_search_tree_kernel_2.h
smart_pointers.o: ../queue.h ../queue/queue_kernel_1.h
smart_pointers.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
smart_pointers.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
smart_pointers.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
smart_pointers.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
smart_pointers.o: ../set/set_kernel_c.h binary_search_tree.h
smart_pointers.o: ../../dlib/memory_manager_global.h
smart_pointers.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
smart_pointers.o: ../memory_manager/memory_manager_kernel_abstract.h
smart_pointers.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
smart_pointers.o: ../../dlib/memory_manager_stateless.h
smart_pointers.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
smart_pointers.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
smart_pointers.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
smart_pointers.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
smart_pointers.o: ../set/set_compare_abstract.h
smart_pointers.o: ../threads/auto_mutex_extension.h
smart_pointers.o: ../threads/auto_unlock_extension.h
smart_pointers.o: ../threads/auto_unlock_extension_abstract.h
smart_pointers.o: ../threads/create_new_thread_extension.h
smart_pointers.o: ../threads/create_new_thread_extension_abstract.h
smart_pointers.o: ../threads/multithreaded_object_extension.h
smart_pointers.o: ../threads/multithreaded_object_extension_abstract.h
smart_pointers.o: ../threads/rsignaler_extension.h
smart_pointers.o: ../threads/rsignaler_extension_abstract.h ../map.h
smart_pointers.o: ../threads/rmutex_extension.h
smart_pointers.o: ../threads/rsignaler_extension.h
smart_pointers.o: ../threads/threaded_object_extension.h
smart_pointers.o: ../threads/threaded_object_extension_abstract.h
smart_pointers.o: ../threads/thread_specific_data_extension.h
smart_pointers.o: ../threads/thread_specific_data_extension_abstract.h
smart_pointers.o: ../threads/thread_function_extension.h
smart_pointers.o: ../threads/thread_function_extension_abstract.h
smart_pointers.o: ../threads/threaded_object_extension.h ../misc_api.h
smart_pointers.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
smart_pointers.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
smart_pointers.o: ../../dlib/logger/logger_kernel_abstract.h
smart_pointers.o: ../smart_pointers.h
smart_pointers.o: ../../dlib/logger/extra_logger_headers.h
smart_pointers.o: ../../dlib/logger/logger_kernel_1.h
smart_pointers.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
smart_pointers.o: ../config_reader/config_reader_kernel_1.h
smart_pointers.o: ../config_reader/config_reader_kernel_abstract.h
smart_pointers.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
smart_pointers.o: ../../dlib/map/map_kernel_abstract.h
smart_pointers.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
smart_pointers.o: ../tokenizer/tokenizer_kernel_1.h
smart_pointers.o: ../tokenizer/tokenizer_kernel_abstract.h
smart_pointers.o: ../tokenizer/tokenizer_kernel_c.h
smart_pointers.o: ../config_reader/config_reader_thread_safe_1.h
smart_pointers.o: ../config_reader/config_reader_thread_safe_abstract.h
smart_pointers.o: ../../dlib/assert.h ../../dlib/algs.h
sockets.o: ../../dlib/sockets.h ../platform.h ../sockets/posix.h
sockets.o: ../sockets/sockets_kernel_2.h ../platform.h
sockets.o: ../sockets/sockets_kernel_abstract.h /usr/include/sys/types.h
sockets.o: /usr/include/features.h /usr/include/sys/cdefs.h
sockets.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
sockets.o: /usr/include/gnu/stubs-32.h /usr/include/bits/types.h
sockets.o: /usr/include/bits/typesizes.h /usr/include/time.h
sockets.o: /usr/include/endian.h /usr/include/bits/endian.h
sockets.o: /usr/include/sys/select.h /usr/include/bits/select.h
sockets.o: /usr/include/bits/sigset.h /usr/include/bits/time.h
sockets.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
sockets.o: /usr/include/sys/socket.h /usr/include/sys/uio.h
sockets.o: /usr/include/bits/uio.h /usr/include/bits/socket.h
sockets.o: /usr/include/limits.h /usr/include/bits/posix1_lim.h
sockets.o: /usr/include/bits/local_lim.h /usr/include/linux/limits.h
sockets.o: /usr/include/bits/posix2_lim.h /usr/include/bits/sockaddr.h
sockets.o: /usr/include/asm/socket.h /usr/include/asm-i386/socket.h
sockets.o: /usr/include/asm/sockios.h /usr/include/asm-i386/sockios.h
sockets.o: /usr/include/errno.h /usr/include/bits/errno.h
sockets.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
sockets.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
sockets.o: /usr/include/asm-generic/errno-base.h /usr/include/arpa/inet.h
sockets.o: /usr/include/netinet/in.h /usr/include/stdint.h
sockets.o: /usr/include/bits/wchar.h /usr/include/bits/in.h
sockets.o: /usr/include/bits/byteswap.h /usr/include/signal.h
sockets.o: /usr/include/inttypes.h /usr/include/netdb.h
sockets.o: /usr/include/rpc/netdb.h /usr/include/bits/netdb.h
sockets.o: /usr/include/unistd.h /usr/include/bits/posix_opt.h
sockets.o: /usr/include/bits/confname.h /usr/include/getopt.h
sockets.o: /usr/include/sys/param.h /usr/include/linux/param.h
sockets.o: /usr/include/asm/param.h /usr/include/asm-i386/param.h
sockets.o: ../threads.h ../threads/threads_kernel.h ../threads/posix.h
sockets.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
sockets.o: /usr/include/pthread.h /usr/include/sched.h
sockets.o: /usr/include/bits/sched.h /usr/include/bits/setjmp.h
sockets.o: /usr/include/sys/time.h ../algs.h ../assert.h ../error.h
sockets.o: ../noncopyable.h ../threads/threads_kernel_shared.h
sockets.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
sockets.o: ../threads/rmutex_extension.h
sockets.o: ../threads/rmutex_extension_abstract.h
sockets.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
sockets.o: ../binary_search_tree/binary_search_tree_kernel_1.h
sockets.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
sockets.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
sockets.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
sockets.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
sockets.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sockets.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
sockets.o: ../../dlib/memory_manager.h ../member_function_pointer.h
sockets.o: ../member_function_pointer/member_function_pointer_kernel_1.h
sockets.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
sockets.o: ../member_function_pointer/member_function_pointer_kernel_c.h
sockets.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
sockets.o: ../memory_manager/memory_manager_kernel_abstract.h
sockets.o: ../memory_manager/memory_manager_kernel_2.h
sockets.o: ../memory_manager/memory_manager_kernel_3.h
sockets.o: ../memory_manager/memory_manager_kernel_2.h
sockets.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
sockets.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
sockets.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
sockets.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
sockets.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
sockets.o: ../set/set_kernel_c.h binary_search_tree.h
sockets.o: ../../dlib/memory_manager_global.h
sockets.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
sockets.o: ../memory_manager/memory_manager_kernel_abstract.h
sockets.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
sockets.o: ../../dlib/memory_manager_stateless.h
sockets.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
sockets.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
sockets.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
sockets.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
sockets.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
sockets.o: ../misc_api.h ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
sockets.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
sockets.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
sockets.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
sockets.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
sockets.o: ../smart_pointers/shared_ptr.h
sockets.o: ../smart_pointers/shared_ptr_abstract.h
sockets.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
sockets.o: ../smart_pointers/weak_ptr_abstract.h
sockets.o: ../../dlib/logger/extra_logger_headers.h
sockets.o: ../../dlib/logger/logger_kernel_1.h
sockets.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
sockets.o: ../config_reader/config_reader_kernel_1.h
sockets.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
sockets.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
sockets.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
sockets.o: ../tokenizer/tokenizer_kernel_1.h
sockets.o: ../tokenizer/tokenizer_kernel_abstract.h
sockets.o: ../tokenizer/tokenizer_kernel_c.h
sockets.o: ../config_reader/config_reader_thread_safe_1.h
sockets.o: ../config_reader/config_reader_thread_safe_abstract.h
sockets.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
sockets.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
sockets.o: ../threads/auto_unlock_extension.h
sockets.o: ../threads/auto_unlock_extension_abstract.h
sockets.o: ../threads/create_new_thread_extension.h
sockets.o: ../threads/create_new_thread_extension_abstract.h
sockets.o: ../threads/multithreaded_object_extension.h
sockets.o: ../threads/multithreaded_object_extension_abstract.h
sockets.o: ../threads/rsignaler_extension.h
sockets.o: ../threads/rsignaler_extension_abstract.h
sockets.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
sockets.o: ../threads/threaded_object_extension.h
sockets.o: ../threads/threaded_object_extension_abstract.h
sockets.o: ../threads/thread_specific_data_extension.h
sockets.o: ../threads/thread_specific_data_extension_abstract.h
sockets.o: ../threads/thread_function_extension.h
sockets.o: ../threads/thread_function_extension_abstract.h
sockets.o: ../threads/threaded_object_extension.h
sockets.o: ../sockets/sockets_extensions.h ../sockets.h
sockets.o: ../sockets/sockets_extensions_abstract.h ../../dlib/server.h
sockets.o: ../../dlib/server/server_kernel_1.h
sockets.o: ../../dlib/server/server_kernel_abstract.h ../logger.h
sockets.o: ../../dlib/server/server_kernel_c.h
sockets.o: ../../dlib/server/server_iostream_1.h
sockets.o: ../../dlib/server/server_iostream_abstract.h
sockets.o: ../../dlib/server/server_http_1.h
sockets.o: ../../dlib/server/server_http_abstract.h ../../dlib/set.h
sockets.o: ../../dlib/sockstreambuf.h
sockets.o: ../../dlib/sockstreambuf/sockstreambuf_kernel_1.h
sockets.o: ../sockstreambuf/sockstreambuf_kernel_abstract.h
sockets.o: ../../dlib/sockstreambuf/sockstreambuf_kernel_2.h
sockets.o: ../../dlib/queue.h ../../dlib/misc_api.h
sockstreambuf.o: ../../dlib/sockets.h ../platform.h ../sockets/posix.h
sockstreambuf.o: ../sockets/sockets_kernel_2.h ../platform.h
sockstreambuf.o: ../sockets/sockets_kernel_abstract.h
sockstreambuf.o: /usr/include/sys/types.h /usr/include/features.h
sockstreambuf.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
sockstreambuf.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
sockstreambuf.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
sockstreambuf.o: /usr/include/time.h /usr/include/endian.h
sockstreambuf.o: /usr/include/bits/endian.h /usr/include/sys/select.h
sockstreambuf.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
sockstreambuf.o: /usr/include/bits/time.h /usr/include/sys/sysmacros.h
sockstreambuf.o: /usr/include/bits/pthreadtypes.h /usr/include/sys/socket.h
sockstreambuf.o: /usr/include/sys/uio.h /usr/include/bits/uio.h
sockstreambuf.o: /usr/include/bits/socket.h /usr/include/limits.h
sockstreambuf.o: /usr/include/bits/posix1_lim.h /usr/include/bits/local_lim.h
sockstreambuf.o: /usr/include/linux/limits.h /usr/include/bits/posix2_lim.h
sockstreambuf.o: /usr/include/bits/sockaddr.h /usr/include/asm/socket.h
sockstreambuf.o: /usr/include/asm-i386/socket.h /usr/include/asm/sockios.h
sockstreambuf.o: /usr/include/asm-i386/sockios.h /usr/include/errno.h
sockstreambuf.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
sockstreambuf.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
sockstreambuf.o: /usr/include/asm-generic/errno.h
sockstreambuf.o: /usr/include/asm-generic/errno-base.h
sockstreambuf.o: /usr/include/arpa/inet.h /usr/include/netinet/in.h
sockstreambuf.o: /usr/include/stdint.h /usr/include/bits/wchar.h
sockstreambuf.o: /usr/include/bits/in.h /usr/include/bits/byteswap.h
sockstreambuf.o: /usr/include/signal.h /usr/include/inttypes.h
sockstreambuf.o: /usr/include/netdb.h /usr/include/rpc/netdb.h
sockstreambuf.o: /usr/include/bits/netdb.h /usr/include/unistd.h
sockstreambuf.o: /usr/include/bits/posix_opt.h /usr/include/bits/confname.h
sockstreambuf.o: /usr/include/getopt.h /usr/include/sys/param.h
sockstreambuf.o: /usr/include/linux/param.h /usr/include/asm/param.h
sockstreambuf.o: /usr/include/asm-i386/param.h ../threads.h
sockstreambuf.o: ../threads/threads_kernel.h ../threads/posix.h
sockstreambuf.o: ../threads/threads_kernel_2.h
sockstreambuf.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
sockstreambuf.o: /usr/include/sched.h /usr/include/bits/sched.h
sockstreambuf.o: /usr/include/bits/setjmp.h /usr/include/sys/time.h ../algs.h
sockstreambuf.o: ../assert.h ../error.h ../noncopyable.h
sockstreambuf.o: ../threads/threads_kernel_shared.h
sockstreambuf.o: ../threads/auto_mutex_extension.h
sockstreambuf.o: ../threads/threads_kernel.h ../threads/rmutex_extension.h
sockstreambuf.o: ../threads/rmutex_extension_abstract.h
sockstreambuf.o: ../threads/auto_mutex_extension_abstract.h
sockstreambuf.o: ../binary_search_tree.h
sockstreambuf.o: ../binary_search_tree/binary_search_tree_kernel_1.h
sockstreambuf.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
sockstreambuf.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
sockstreambuf.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
sockstreambuf.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
sockstreambuf.o: ../enable_if.h
sockstreambuf.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sockstreambuf.o: ../binary_search_tree/binary_search_tree_kernel_c.h
sockstreambuf.o: ../assert.h ../../dlib/memory_manager.h
sockstreambuf.o: ../member_function_pointer.h
sockstreambuf.o: ../member_function_pointer/member_function_pointer_kernel_1.h
sockstreambuf.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
sockstreambuf.o: ../member_function_pointer/member_function_pointer_kernel_c.h
sockstreambuf.o: ../memory_manager.h
sockstreambuf.o: ../memory_manager/memory_manager_kernel_1.h
sockstreambuf.o: ../memory_manager/memory_manager_kernel_abstract.h
sockstreambuf.o: ../memory_manager/memory_manager_kernel_2.h
sockstreambuf.o: ../memory_manager/memory_manager_kernel_3.h
sockstreambuf.o: ../memory_manager/memory_manager_kernel_2.h
sockstreambuf.o: ../binary_search_tree/binary_search_tree_kernel_2.h
sockstreambuf.o: ../queue.h ../queue/queue_kernel_1.h
sockstreambuf.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
sockstreambuf.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
sockstreambuf.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
sockstreambuf.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
sockstreambuf.o: ../set/set_kernel_c.h binary_search_tree.h
sockstreambuf.o: ../../dlib/memory_manager_global.h
sockstreambuf.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
sockstreambuf.o: ../memory_manager/memory_manager_kernel_abstract.h
sockstreambuf.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
sockstreambuf.o: ../../dlib/memory_manager_stateless.h
sockstreambuf.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
sockstreambuf.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
sockstreambuf.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
sockstreambuf.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
sockstreambuf.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
sockstreambuf.o: ../misc_api.h ../misc_api/posix.h
sockstreambuf.o: ../misc_api/misc_api_kernel_2.h
sockstreambuf.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
sockstreambuf.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
sockstreambuf.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h
sockstreambuf.o: ../noncopyable.h ../smart_pointers/scoped_ptr_abstract.h
sockstreambuf.o: ../smart_pointers/shared_ptr.h
sockstreambuf.o: ../smart_pointers/shared_ptr_abstract.h
sockstreambuf.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
sockstreambuf.o: ../smart_pointers/weak_ptr_abstract.h
sockstreambuf.o: ../../dlib/logger/extra_logger_headers.h
sockstreambuf.o: ../../dlib/logger/logger_kernel_1.h
sockstreambuf.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
sockstreambuf.o: ../config_reader/config_reader_kernel_1.h
sockstreambuf.o: ../config_reader/config_reader_kernel_abstract.h
sockstreambuf.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
sockstreambuf.o: ../../dlib/map/map_kernel_abstract.h
sockstreambuf.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
sockstreambuf.o: ../tokenizer/tokenizer_kernel_1.h
sockstreambuf.o: ../tokenizer/tokenizer_kernel_abstract.h
sockstreambuf.o: ../tokenizer/tokenizer_kernel_c.h
sockstreambuf.o: ../config_reader/config_reader_thread_safe_1.h
sockstreambuf.o: ../config_reader/config_reader_thread_safe_abstract.h
sockstreambuf.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
sockstreambuf.o: ../set/set_compare_abstract.h
sockstreambuf.o: ../threads/auto_mutex_extension.h
sockstreambuf.o: ../threads/auto_unlock_extension.h
sockstreambuf.o: ../threads/auto_unlock_extension_abstract.h
sockstreambuf.o: ../threads/create_new_thread_extension.h
sockstreambuf.o: ../threads/create_new_thread_extension_abstract.h
sockstreambuf.o: ../threads/multithreaded_object_extension.h
sockstreambuf.o: ../threads/multithreaded_object_extension_abstract.h
sockstreambuf.o: ../threads/rsignaler_extension.h
sockstreambuf.o: ../threads/rsignaler_extension_abstract.h
sockstreambuf.o: ../threads/rmutex_extension.h
sockstreambuf.o: ../threads/rsignaler_extension.h
sockstreambuf.o: ../threads/threaded_object_extension.h
sockstreambuf.o: ../threads/threaded_object_extension_abstract.h
sockstreambuf.o: ../threads/thread_specific_data_extension.h
sockstreambuf.o: ../threads/thread_specific_data_extension_abstract.h
sockstreambuf.o: ../threads/thread_function_extension.h
sockstreambuf.o: ../threads/thread_function_extension_abstract.h
sockstreambuf.o: ../threads/threaded_object_extension.h
sockstreambuf.o: ../sockets/sockets_extensions.h ../sockets.h
sockstreambuf.o: ../sockets/sockets_extensions_abstract.h
sockstreambuf.o: ../../dlib/misc_api.h ../../dlib/sockstreambuf.h
sockstreambuf.o: ../../dlib/sockstreambuf/sockstreambuf_kernel_1.h
sockstreambuf.o: ../sockstreambuf/sockstreambuf_kernel_abstract.h
sockstreambuf.o: ../../dlib/sockstreambuf/sockstreambuf_kernel_2.h
sockstreambuf.o: ../../dlib/smart_pointers.h
stack.o: ../../dlib/stack.h ../../dlib/stack/stack_kernel_1.h
stack.o: ../../dlib/stack/stack_kernel_abstract.h ../algs.h ../platform.h
stack.o: ../assert.h ../error.h ../noncopyable.h ../interfaces/enumerable.h
stack.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
stack.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
stack.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
stack.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
stack.o: ../memory_manager/memory_manager_kernel_2.h
stack.o: ../memory_manager/memory_manager_kernel_3.h
stack.o: ../memory_manager/memory_manager_kernel_2.h
stack.o: ../binary_search_tree/binary_search_tree_kernel_2.h
stack.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
stack.o: ../interfaces/map_pair.h ../../dlib/stack/stack_kernel_c.h
stack.o: ../../dlib/memory_manager.h tester.h ../../dlib/map.h
stack.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../threads.h
stack.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
stack.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
stack.o: /usr/include/pthread.h /usr/include/features.h
stack.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
stack.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
stack.o: /usr/include/sched.h /usr/include/bits/types.h
stack.o: /usr/include/bits/typesizes.h /usr/include/time.h
stack.o: /usr/include/bits/sched.h /usr/include/signal.h
stack.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
stack.o: /usr/include/bits/setjmp.h /usr/include/errno.h
stack.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
stack.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
stack.o: /usr/include/asm-generic/errno.h
stack.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
stack.o: /usr/include/bits/time.h /usr/include/sys/select.h
stack.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
stack.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
stack.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
stack.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
stack.o: ../binary_search_tree/binary_search_tree_kernel_1.h
stack.o: ../binary_search_tree/binary_search_tree_kernel_2.h
stack.o: ../binary_search_tree/binary_search_tree_kernel_c.h
stack.o: ../member_function_pointer.h
stack.o: ../member_function_pointer/member_function_pointer_kernel_1.h
stack.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
stack.o: ../member_function_pointer/member_function_pointer_kernel_c.h
stack.o: ../queue.h ../queue/queue_kernel_1.h
stack.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
stack.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
stack.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
stack.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
stack.o: ../set/set_kernel_c.h binary_search_tree.h
stack.o: ../../dlib/memory_manager_global.h
stack.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
stack.o: ../memory_manager/memory_manager_kernel_abstract.h
stack.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
stack.o: ../../dlib/memory_manager_stateless.h
stack.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
stack.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
stack.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
stack.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
stack.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
stack.o: ../threads/auto_unlock_extension.h
stack.o: ../threads/auto_unlock_extension_abstract.h
stack.o: ../threads/create_new_thread_extension.h
stack.o: ../threads/create_new_thread_extension_abstract.h
stack.o: ../threads/multithreaded_object_extension.h
stack.o: ../threads/multithreaded_object_extension_abstract.h
stack.o: ../threads/rsignaler_extension.h
stack.o: ../threads/rsignaler_extension_abstract.h ../map.h
stack.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
stack.o: ../threads/threaded_object_extension.h
stack.o: ../threads/threaded_object_extension_abstract.h
stack.o: ../threads/thread_specific_data_extension.h
stack.o: ../threads/thread_specific_data_extension_abstract.h
stack.o: ../threads/thread_function_extension.h
stack.o: ../threads/thread_function_extension_abstract.h
stack.o: ../threads/threaded_object_extension.h ../misc_api.h
stack.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
stack.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
stack.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
stack.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
stack.o: ../smart_pointers/scoped_ptr_abstract.h
stack.o: ../smart_pointers/shared_ptr.h
stack.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
stack.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
stack.o: ../../dlib/logger/extra_logger_headers.h
stack.o: ../../dlib/logger/logger_kernel_1.h
stack.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
stack.o: ../config_reader/config_reader_kernel_1.h
stack.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
stack.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
stack.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
stack.o: ../tokenizer/tokenizer_kernel_1.h
stack.o: ../tokenizer/tokenizer_kernel_abstract.h
stack.o: ../tokenizer/tokenizer_kernel_c.h
stack.o: ../config_reader/config_reader_thread_safe_1.h
stack.o: ../config_reader/config_reader_thread_safe_abstract.h
stack.o: ../../dlib/assert.h ../../dlib/algs.h
static_map.o: ../../dlib/hash_table.h
static_map.o: ../../dlib/hash_table/hash_table_kernel_1.h
static_map.o: ../../dlib/hash_table/hash_table_kernel_abstract.h
static_map.o: ../general_hash/general_hash.h ../algs.h ../platform.h
static_map.o: ../assert.h ../error.h ../noncopyable.h
static_map.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
static_map.o: ../interfaces/remover.h ../assert.h ../serialize.h ../algs.h
static_map.o: ../uintn.h ../interfaces/enumerable.h ../interfaces/map_pair.h
static_map.o: ../enable_if.h ../memory_manager.h
static_map.o: ../memory_manager/memory_manager_kernel_1.h
static_map.o: ../memory_manager/memory_manager_kernel_abstract.h
static_map.o: ../memory_manager/memory_manager_kernel_2.h
static_map.o: ../memory_manager/memory_manager_kernel_3.h
static_map.o: ../memory_manager/memory_manager_kernel_2.h
static_map.o: ../binary_search_tree/binary_search_tree_kernel_2.h
static_map.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
static_map.o: ../../dlib/hash_table/hash_table_kernel_2.h
static_map.o: ../../dlib/hash_table/hash_table_kernel_c.h
static_map.o: ../../dlib/memory_manager.h binary_search_tree.h
static_map.o: ../../dlib/memory_manager_global.h
static_map.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
static_map.o: ../memory_manager/memory_manager_kernel_abstract.h
static_map.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
static_map.o: ../../dlib/memory_manager_stateless.h
static_map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
static_map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
static_map.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
static_map.o: ../threads.h ../threads/threads_kernel.h ../platform.h
static_map.o: ../threads/posix.h ../threads/threads_kernel_2.h
static_map.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
static_map.o: /usr/include/features.h /usr/include/sys/cdefs.h
static_map.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
static_map.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
static_map.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
static_map.o: /usr/include/time.h /usr/include/bits/sched.h
static_map.o: /usr/include/signal.h /usr/include/bits/sigset.h
static_map.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
static_map.o: /usr/include/errno.h /usr/include/bits/errno.h
static_map.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
static_map.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
static_map.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
static_map.o: /usr/include/bits/time.h /usr/include/sys/select.h
static_map.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
static_map.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
static_map.o: ../threads/rmutex_extension.h
static_map.o: ../threads/rmutex_extension_abstract.h
static_map.o: ../threads/auto_mutex_extension_abstract.h
static_map.o: ../binary_search_tree.h
static_map.o: ../binary_search_tree/binary_search_tree_kernel_1.h
static_map.o: ../binary_search_tree/binary_search_tree_kernel_2.h
static_map.o: ../binary_search_tree/binary_search_tree_kernel_c.h
static_map.o: ../member_function_pointer.h
static_map.o: ../member_function_pointer/member_function_pointer_kernel_1.h
static_map.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
static_map.o: ../member_function_pointer/member_function_pointer_kernel_c.h
static_map.o: ../queue.h ../queue/queue_kernel_1.h
static_map.o: ../queue/queue_kernel_abstract.h ../queue/queue_kernel_2.h
static_map.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
static_map.o: ../queue/queue_sort_abstract.h ../sort.h ../set.h
static_map.o: ../set/set_kernel_1.h ../set/set_kernel_abstract.h
static_map.o: ../set/set_kernel_c.h ../set/set_compare_1.h
static_map.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
static_map.o: ../threads/auto_unlock_extension.h
static_map.o: ../threads/auto_unlock_extension_abstract.h
static_map.o: ../threads/create_new_thread_extension.h
static_map.o: ../threads/create_new_thread_extension_abstract.h
static_map.o: ../threads/multithreaded_object_extension.h
static_map.o: ../threads/multithreaded_object_extension_abstract.h
static_map.o: ../threads/rsignaler_extension.h
static_map.o: ../threads/rsignaler_extension_abstract.h ../map.h
static_map.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
static_map.o: ../threads/threaded_object_extension.h
static_map.o: ../threads/threaded_object_extension_abstract.h
static_map.o: ../threads/thread_specific_data_extension.h
static_map.o: ../threads/thread_specific_data_extension_abstract.h
static_map.o: ../threads/thread_function_extension.h
static_map.o: ../threads/thread_function_extension_abstract.h
static_map.o: ../threads/threaded_object_extension.h
static_map.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
static_map.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
static_map.o: ../misc_api.h ../misc_api/posix.h
static_map.o: ../misc_api/misc_api_kernel_2.h
static_map.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
static_map.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
static_map.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
static_map.o: ../smart_pointers/scoped_ptr_abstract.h
static_map.o: ../smart_pointers/shared_ptr.h
static_map.o: ../smart_pointers/shared_ptr_abstract.h
static_map.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
static_map.o: ../smart_pointers/weak_ptr_abstract.h
static_map.o: ../../dlib/logger/extra_logger_headers.h
static_map.o: ../../dlib/logger/logger_kernel_1.h
static_map.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
static_map.o: ../config_reader/config_reader_kernel_1.h
static_map.o: ../config_reader/config_reader_kernel_abstract.h
static_map.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
static_map.o: ../../dlib/map/map_kernel_abstract.h
static_map.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
static_map.o: ../tokenizer/tokenizer_kernel_1.h
static_map.o: ../tokenizer/tokenizer_kernel_abstract.h
static_map.o: ../tokenizer/tokenizer_kernel_c.h
static_map.o: ../config_reader/config_reader_thread_safe_1.h
static_map.o: ../config_reader/config_reader_thread_safe_abstract.h
static_map.o: ../../dlib/assert.h ../../dlib/algs.h ../../dlib/static_map.h
static_map.o: ../../dlib/static_map/static_map_kernel_1.h
static_map.o: ../../dlib/static_map/static_map_kernel_abstract.h
static_map.o: ../../dlib/static_map/static_map_kernel_c.h
static_set.o: ../../dlib/queue.h ../queue/queue_kernel_1.h
static_set.o: ../queue/queue_kernel_abstract.h ../algs.h ../platform.h
static_set.o: ../assert.h ../error.h ../noncopyable.h
static_set.o: ../interfaces/enumerable.h ../interfaces/remover.h
static_set.o: ../serialize.h ../algs.h ../uintn.h ../interfaces/enumerable.h
static_set.o: ../interfaces/map_pair.h ../enable_if.h ../memory_manager.h
static_set.o: ../memory_manager/memory_manager_kernel_1.h
static_set.o: ../memory_manager/memory_manager_kernel_abstract.h ../assert.h
static_set.o: ../memory_manager/memory_manager_kernel_2.h
static_set.o: ../memory_manager/memory_manager_kernel_3.h
static_set.o: ../memory_manager/memory_manager_kernel_2.h
static_set.o: ../binary_search_tree/binary_search_tree_kernel_2.h
static_set.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
static_set.o: ../interfaces/map_pair.h ../queue/queue_kernel_2.h
static_set.o: ../queue/queue_kernel_c.h ../queue/queue_sort_1.h
static_set.o: ../queue/queue_sort_abstract.h ../sort.h
static_set.o: ../../dlib/memory_manager.h ../../dlib/static_set.h
static_set.o: ../../dlib/static_set/static_set_kernel_1.h
static_set.o: ../../dlib/static_set/static_set_kernel_abstract.h
static_set.o: ../../dlib/static_set/static_set_kernel_c.h
static_set.o: ../../dlib/static_set/static_set_compare_1.h
static_set.o: ../../dlib/static_set/static_set_compare_abstract.h
static_set.o: ../../dlib/set.h tester.h ../../dlib/map.h ../../dlib/logger.h
static_set.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
static_set.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
static_set.o: ../threads/threads_kernel_2.h
static_set.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
static_set.o: /usr/include/features.h /usr/include/sys/cdefs.h
static_set.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
static_set.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
static_set.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
static_set.o: /usr/include/time.h /usr/include/bits/sched.h
static_set.o: /usr/include/signal.h /usr/include/bits/sigset.h
static_set.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
static_set.o: /usr/include/errno.h /usr/include/bits/errno.h
static_set.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
static_set.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
static_set.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
static_set.o: /usr/include/bits/time.h /usr/include/sys/select.h
static_set.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
static_set.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
static_set.o: ../threads/rmutex_extension.h
static_set.o: ../threads/rmutex_extension_abstract.h
static_set.o: ../threads/auto_mutex_extension_abstract.h
static_set.o: ../binary_search_tree.h
static_set.o: ../binary_search_tree/binary_search_tree_kernel_1.h
static_set.o: ../binary_search_tree/binary_search_tree_kernel_2.h
static_set.o: ../binary_search_tree/binary_search_tree_kernel_c.h
static_set.o: ../member_function_pointer.h
static_set.o: ../member_function_pointer/member_function_pointer_kernel_1.h
static_set.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
static_set.o: ../member_function_pointer/member_function_pointer_kernel_c.h
static_set.o: ../queue.h ../set.h ../set/set_kernel_1.h
static_set.o: ../set/set_kernel_abstract.h ../set/set_kernel_c.h
static_set.o: binary_search_tree.h ../../dlib/memory_manager_global.h
static_set.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
static_set.o: ../memory_manager/memory_manager_kernel_abstract.h
static_set.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
static_set.o: ../../dlib/memory_manager_stateless.h
static_set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
static_set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
static_set.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
static_set.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
static_set.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
static_set.o: ../threads/auto_unlock_extension.h
static_set.o: ../threads/auto_unlock_extension_abstract.h
static_set.o: ../threads/create_new_thread_extension.h
static_set.o: ../threads/create_new_thread_extension_abstract.h
static_set.o: ../threads/multithreaded_object_extension.h
static_set.o: ../threads/multithreaded_object_extension_abstract.h
static_set.o: ../threads/rsignaler_extension.h
static_set.o: ../threads/rsignaler_extension_abstract.h ../map.h
static_set.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
static_set.o: ../threads/threaded_object_extension.h
static_set.o: ../threads/threaded_object_extension_abstract.h
static_set.o: ../threads/thread_specific_data_extension.h
static_set.o: ../threads/thread_specific_data_extension_abstract.h
static_set.o: ../threads/thread_function_extension.h
static_set.o: ../threads/thread_function_extension_abstract.h
static_set.o: ../threads/threaded_object_extension.h ../misc_api.h
static_set.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
static_set.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
static_set.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
static_set.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
static_set.o: ../smart_pointers/scoped_ptr_abstract.h
static_set.o: ../smart_pointers/shared_ptr.h
static_set.o: ../smart_pointers/shared_ptr_abstract.h
static_set.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
static_set.o: ../smart_pointers/weak_ptr_abstract.h
static_set.o: ../../dlib/logger/extra_logger_headers.h
static_set.o: ../../dlib/logger/logger_kernel_1.h
static_set.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
static_set.o: ../config_reader/config_reader_kernel_1.h
static_set.o: ../config_reader/config_reader_kernel_abstract.h
static_set.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
static_set.o: ../../dlib/map/map_kernel_abstract.h
static_set.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
static_set.o: ../tokenizer/tokenizer_kernel_1.h
static_set.o: ../tokenizer/tokenizer_kernel_abstract.h
static_set.o: ../tokenizer/tokenizer_kernel_c.h
static_set.o: ../config_reader/config_reader_thread_safe_1.h
static_set.o: ../config_reader/config_reader_thread_safe_abstract.h
static_set.o: ../../dlib/assert.h ../../dlib/algs.h
string.o: ../../dlib/string.h tester.h ../../dlib/map.h ../../dlib/logger.h
string.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
string.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
string.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
string.o: /usr/include/pthread.h /usr/include/features.h
string.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
string.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
string.o: /usr/include/sched.h /usr/include/bits/types.h
string.o: /usr/include/bits/typesizes.h /usr/include/time.h
string.o: /usr/include/bits/sched.h /usr/include/signal.h
string.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
string.o: /usr/include/bits/setjmp.h /usr/include/errno.h
string.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
string.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
string.o: /usr/include/asm-generic/errno.h
string.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
string.o: /usr/include/bits/time.h /usr/include/sys/select.h
string.o: /usr/include/bits/select.h ../algs.h ../platform.h ../assert.h
string.o: ../error.h ../noncopyable.h ../threads/threads_kernel_shared.h
string.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
string.o: ../threads/rmutex_extension.h
string.o: ../threads/rmutex_extension_abstract.h
string.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
string.o: ../binary_search_tree/binary_search_tree_kernel_1.h
string.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
string.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
string.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
string.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
string.o: ../binary_search_tree/binary_search_tree_kernel_2.h
string.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
string.o: ../../dlib/memory_manager.h ../member_function_pointer.h
string.o: ../member_function_pointer/member_function_pointer_kernel_1.h
string.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
string.o: ../member_function_pointer/member_function_pointer_kernel_c.h
string.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
string.o: ../memory_manager/memory_manager_kernel_abstract.h
string.o: ../memory_manager/memory_manager_kernel_2.h
string.o: ../memory_manager/memory_manager_kernel_3.h
string.o: ../memory_manager/memory_manager_kernel_2.h
string.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
string.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
string.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
string.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
string.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
string.o: ../set/set_kernel_c.h binary_search_tree.h
string.o: ../../dlib/memory_manager_global.h
string.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
string.o: ../memory_manager/memory_manager_kernel_abstract.h
string.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
string.o: ../../dlib/memory_manager_stateless.h
string.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
string.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
string.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
string.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
string.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
string.o: ../threads/auto_unlock_extension.h
string.o: ../threads/auto_unlock_extension_abstract.h
string.o: ../threads/create_new_thread_extension.h
string.o: ../threads/create_new_thread_extension_abstract.h
string.o: ../threads/multithreaded_object_extension.h
string.o: ../threads/multithreaded_object_extension_abstract.h
string.o: ../threads/rsignaler_extension.h
string.o: ../threads/rsignaler_extension_abstract.h ../map.h
string.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
string.o: ../threads/threaded_object_extension.h
string.o: ../threads/threaded_object_extension_abstract.h
string.o: ../threads/thread_specific_data_extension.h
string.o: ../threads/thread_specific_data_extension_abstract.h
string.o: ../threads/thread_function_extension.h
string.o: ../threads/thread_function_extension_abstract.h
string.o: ../threads/threaded_object_extension.h ../misc_api.h
string.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
string.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
string.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
string.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
string.o: ../smart_pointers/scoped_ptr_abstract.h
string.o: ../smart_pointers/shared_ptr.h
string.o: ../smart_pointers/shared_ptr_abstract.h
string.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
string.o: ../smart_pointers/weak_ptr_abstract.h
string.o: ../../dlib/logger/extra_logger_headers.h
string.o: ../../dlib/logger/logger_kernel_1.h
string.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
string.o: ../config_reader/config_reader_kernel_1.h
string.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
string.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
string.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
string.o: ../tokenizer/tokenizer_kernel_1.h
string.o: ../tokenizer/tokenizer_kernel_abstract.h
string.o: ../tokenizer/tokenizer_kernel_c.h
string.o: ../config_reader/config_reader_thread_safe_1.h
string.o: ../config_reader/config_reader_thread_safe_abstract.h
string.o: ../../dlib/assert.h ../../dlib/algs.h
threads.o: ../../dlib/misc_api.h ../platform.h ../misc_api/posix.h
threads.o: ../misc_api/misc_api_kernel_2.h
threads.o: ../misc_api/misc_api_kernel_abstract.h ../algs.h ../assert.h
threads.o: ../error.h ../noncopyable.h ../uintn.h ../../dlib/threads.h
threads.o: tester.h ../../dlib/map.h ../../dlib/logger.h
threads.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
threads.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
threads.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
threads.o: /usr/include/pthread.h /usr/include/features.h
threads.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
threads.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
threads.o: /usr/include/sched.h /usr/include/bits/types.h
threads.o: /usr/include/bits/typesizes.h /usr/include/time.h
threads.o: /usr/include/bits/sched.h /usr/include/signal.h
threads.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
threads.o: /usr/include/bits/setjmp.h /usr/include/errno.h
threads.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
threads.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
threads.o: /usr/include/asm-generic/errno.h
threads.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
threads.o: /usr/include/bits/time.h /usr/include/sys/select.h
threads.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
threads.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
threads.o: ../threads/rmutex_extension.h
threads.o: ../threads/rmutex_extension_abstract.h
threads.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
threads.o: ../binary_search_tree/binary_search_tree_kernel_1.h
threads.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
threads.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
threads.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
threads.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
threads.o: ../binary_search_tree/binary_search_tree_kernel_2.h
threads.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
threads.o: ../../dlib/memory_manager.h ../member_function_pointer.h
threads.o: ../member_function_pointer/member_function_pointer_kernel_1.h
threads.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
threads.o: ../member_function_pointer/member_function_pointer_kernel_c.h
threads.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
threads.o: ../memory_manager/memory_manager_kernel_abstract.h
threads.o: ../memory_manager/memory_manager_kernel_2.h
threads.o: ../memory_manager/memory_manager_kernel_3.h
threads.o: ../memory_manager/memory_manager_kernel_2.h
threads.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
threads.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
threads.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
threads.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
threads.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
threads.o: ../set/set_kernel_c.h binary_search_tree.h
threads.o: ../../dlib/memory_manager_global.h
threads.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
threads.o: ../memory_manager/memory_manager_kernel_abstract.h
threads.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
threads.o: ../../dlib/memory_manager_stateless.h
threads.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
threads.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
threads.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
threads.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
threads.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
threads.o: ../threads/auto_unlock_extension.h
threads.o: ../threads/auto_unlock_extension_abstract.h
threads.o: ../threads/create_new_thread_extension.h
threads.o: ../threads/create_new_thread_extension_abstract.h
threads.o: ../threads/multithreaded_object_extension.h
threads.o: ../threads/multithreaded_object_extension_abstract.h
threads.o: ../threads/rsignaler_extension.h
threads.o: ../threads/rsignaler_extension_abstract.h ../map.h
threads.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
threads.o: ../threads/threaded_object_extension.h
threads.o: ../threads/threaded_object_extension_abstract.h
threads.o: ../threads/thread_specific_data_extension.h
threads.o: ../threads/thread_specific_data_extension_abstract.h
threads.o: ../threads/thread_function_extension.h
threads.o: ../threads/thread_function_extension_abstract.h
threads.o: ../threads/threaded_object_extension.h ../misc_api.h
threads.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
threads.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
threads.o: ../smart_pointers/scoped_ptr_abstract.h
threads.o: ../smart_pointers/shared_ptr.h
threads.o: ../smart_pointers/shared_ptr_abstract.h
threads.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
threads.o: ../smart_pointers/weak_ptr_abstract.h
threads.o: ../../dlib/logger/extra_logger_headers.h
threads.o: ../../dlib/logger/logger_kernel_1.h
threads.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
threads.o: ../config_reader/config_reader_kernel_1.h
threads.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
threads.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
threads.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
threads.o: ../tokenizer/tokenizer_kernel_1.h
threads.o: ../tokenizer/tokenizer_kernel_abstract.h
threads.o: ../tokenizer/tokenizer_kernel_c.h
threads.o: ../config_reader/config_reader_thread_safe_1.h
threads.o: ../config_reader/config_reader_thread_safe_abstract.h
threads.o: ../../dlib/assert.h ../../dlib/algs.h
timer.o: ../../dlib/timer.h ../timer/timer_kernel_1.h ../threads.h
timer.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
timer.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
timer.o: /usr/include/pthread.h /usr/include/features.h
timer.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
timer.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
timer.o: /usr/include/sched.h /usr/include/bits/types.h
timer.o: /usr/include/bits/typesizes.h /usr/include/time.h
timer.o: /usr/include/bits/sched.h /usr/include/signal.h
timer.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
timer.o: /usr/include/bits/setjmp.h /usr/include/errno.h
timer.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
timer.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
timer.o: /usr/include/asm-generic/errno.h
timer.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
timer.o: /usr/include/bits/time.h /usr/include/sys/select.h
timer.o: /usr/include/bits/select.h ../algs.h ../platform.h ../assert.h
timer.o: ../error.h ../noncopyable.h ../threads/threads_kernel_shared.h
timer.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
timer.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
timer.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
timer.o: ../binary_search_tree/binary_search_tree_kernel_1.h
timer.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
timer.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
timer.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
timer.o: ../interfaces/enumerable.h ../interfaces/map_pair.h ../enable_if.h
timer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
timer.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
timer.o: ../../dlib/memory_manager.h ../member_function_pointer.h
timer.o: ../member_function_pointer/member_function_pointer_kernel_1.h
timer.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
timer.o: ../member_function_pointer/member_function_pointer_kernel_c.h
timer.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
timer.o: ../memory_manager/memory_manager_kernel_abstract.h
timer.o: ../memory_manager/memory_manager_kernel_2.h
timer.o: ../memory_manager/memory_manager_kernel_3.h
timer.o: ../memory_manager/memory_manager_kernel_2.h
timer.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
timer.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
timer.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
timer.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
timer.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
timer.o: ../set/set_kernel_c.h binary_search_tree.h
timer.o: ../../dlib/memory_manager_global.h
timer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
timer.o: ../memory_manager/memory_manager_kernel_abstract.h
timer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
timer.o: ../../dlib/memory_manager_stateless.h
timer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
timer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
timer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
timer.o: ../../dlib/binary_search_tree.h tester.h ../../dlib/map.h
timer.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h
timer.o: ../misc_api.h ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
timer.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
timer.o: ../../dlib/logger/logger_kernel_abstract.h ../map.h
timer.o: ../smart_pointers.h ../smart_pointers/scoped_ptr.h ../noncopyable.h
timer.o: ../smart_pointers/scoped_ptr_abstract.h
timer.o: ../smart_pointers/shared_ptr.h
timer.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
timer.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
timer.o: ../../dlib/logger/extra_logger_headers.h
timer.o: ../../dlib/logger/logger_kernel_1.h
timer.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
timer.o: ../config_reader/config_reader_kernel_1.h
timer.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
timer.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
timer.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
timer.o: ../tokenizer/tokenizer_kernel_1.h
timer.o: ../tokenizer/tokenizer_kernel_abstract.h
timer.o: ../tokenizer/tokenizer_kernel_c.h
timer.o: ../config_reader/config_reader_thread_safe_1.h
timer.o: ../config_reader/config_reader_thread_safe_abstract.h
timer.o: ../../dlib/assert.h ../../dlib/algs.h ../set/set_compare_1.h
timer.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
timer.o: ../threads/auto_unlock_extension.h
timer.o: ../threads/auto_unlock_extension_abstract.h
timer.o: ../threads/create_new_thread_extension.h
timer.o: ../threads/create_new_thread_extension_abstract.h
timer.o: ../threads/multithreaded_object_extension.h
timer.o: ../threads/multithreaded_object_extension_abstract.h
timer.o: ../threads/rsignaler_extension.h
timer.o: ../threads/rsignaler_extension_abstract.h
timer.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
timer.o: ../threads/threaded_object_extension.h
timer.o: ../threads/threaded_object_extension_abstract.h
timer.o: ../threads/thread_specific_data_extension.h
timer.o: ../threads/thread_specific_data_extension_abstract.h
timer.o: ../threads/thread_function_extension.h
timer.o: ../threads/thread_function_extension_abstract.h
timer.o: ../threads/threaded_object_extension.h
timer.o: ../timer/timer_kernel_abstract.h ../timer/timer_kernel_2.h
timer.o: ../../dlib/timeout.h ../timeout/timeout_kernel_1.h
timer.o: ../timeout/timeout_kernel_abstract.h ../timer.h
tokenizer.o: ../../dlib/tokenizer.h ../tokenizer/tokenizer_kernel_1.h
tokenizer.o: ../algs.h ../platform.h ../assert.h ../error.h ../noncopyable.h
tokenizer.o: ../tokenizer/tokenizer_kernel_abstract.h
tokenizer.o: ../tokenizer/tokenizer_kernel_c.h ../assert.h tester.h
tokenizer.o: ../../dlib/map.h ../../dlib/logger.h
tokenizer.o: ../../dlib/logger/logger_kernel_1.h ../threads.h
tokenizer.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
tokenizer.o: ../threads/threads_kernel_2.h
tokenizer.o: ../threads/threads_kernel_abstract.h /usr/include/pthread.h
tokenizer.o: /usr/include/features.h /usr/include/sys/cdefs.h
tokenizer.o: /usr/include/bits/wordsize.h /usr/include/gnu/stubs.h
tokenizer.o: /usr/include/gnu/stubs-32.h /usr/include/sched.h
tokenizer.o: /usr/include/bits/types.h /usr/include/bits/typesizes.h
tokenizer.o: /usr/include/time.h /usr/include/bits/sched.h
tokenizer.o: /usr/include/signal.h /usr/include/bits/sigset.h
tokenizer.o: /usr/include/bits/pthreadtypes.h /usr/include/bits/setjmp.h
tokenizer.o: /usr/include/errno.h /usr/include/bits/errno.h
tokenizer.o: /usr/include/linux/errno.h /usr/include/asm/errno.h
tokenizer.o: /usr/include/asm-i386/errno.h /usr/include/asm-generic/errno.h
tokenizer.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
tokenizer.o: /usr/include/bits/time.h /usr/include/sys/select.h
tokenizer.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
tokenizer.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
tokenizer.o: ../threads/rmutex_extension.h
tokenizer.o: ../threads/rmutex_extension_abstract.h
tokenizer.o: ../threads/auto_mutex_extension_abstract.h
tokenizer.o: ../binary_search_tree.h
tokenizer.o: ../binary_search_tree/binary_search_tree_kernel_1.h
tokenizer.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
tokenizer.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
tokenizer.o: ../interfaces/remover.h ../serialize.h ../algs.h ../uintn.h
tokenizer.o: ../interfaces/enumerable.h ../interfaces/map_pair.h
tokenizer.o: ../enable_if.h
tokenizer.o: ../binary_search_tree/binary_search_tree_kernel_2.h
tokenizer.o: ../binary_search_tree/binary_search_tree_kernel_c.h
tokenizer.o: ../../dlib/memory_manager.h ../member_function_pointer.h
tokenizer.o: ../member_function_pointer/member_function_pointer_kernel_1.h
tokenizer.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
tokenizer.o: ../member_function_pointer/member_function_pointer_kernel_c.h
tokenizer.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
tokenizer.o: ../memory_manager/memory_manager_kernel_abstract.h
tokenizer.o: ../memory_manager/memory_manager_kernel_2.h
tokenizer.o: ../memory_manager/memory_manager_kernel_3.h
tokenizer.o: ../memory_manager/memory_manager_kernel_2.h
tokenizer.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
tokenizer.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
tokenizer.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
tokenizer.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
tokenizer.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
tokenizer.o: ../set/set_kernel_c.h binary_search_tree.h
tokenizer.o: ../../dlib/memory_manager_global.h
tokenizer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
tokenizer.o: ../memory_manager/memory_manager_kernel_abstract.h
tokenizer.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
tokenizer.o: ../../dlib/memory_manager_stateless.h
tokenizer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
tokenizer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
tokenizer.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
tokenizer.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
tokenizer.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
tokenizer.o: ../threads/auto_unlock_extension.h
tokenizer.o: ../threads/auto_unlock_extension_abstract.h
tokenizer.o: ../threads/create_new_thread_extension.h
tokenizer.o: ../threads/create_new_thread_extension_abstract.h
tokenizer.o: ../threads/multithreaded_object_extension.h
tokenizer.o: ../threads/multithreaded_object_extension_abstract.h
tokenizer.o: ../threads/rsignaler_extension.h
tokenizer.o: ../threads/rsignaler_extension_abstract.h ../map.h
tokenizer.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
tokenizer.o: ../threads/threaded_object_extension.h
tokenizer.o: ../threads/threaded_object_extension_abstract.h
tokenizer.o: ../threads/thread_specific_data_extension.h
tokenizer.o: ../threads/thread_specific_data_extension_abstract.h
tokenizer.o: ../threads/thread_function_extension.h
tokenizer.o: ../threads/thread_function_extension_abstract.h
tokenizer.o: ../threads/threaded_object_extension.h ../misc_api.h
tokenizer.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
tokenizer.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
tokenizer.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
tokenizer.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
tokenizer.o: ../smart_pointers/scoped_ptr_abstract.h
tokenizer.o: ../smart_pointers/shared_ptr.h
tokenizer.o: ../smart_pointers/shared_ptr_abstract.h
tokenizer.o: ../smart_pointers/weak_ptr.h ../smart_pointers/shared_ptr.h
tokenizer.o: ../smart_pointers/weak_ptr_abstract.h
tokenizer.o: ../../dlib/logger/extra_logger_headers.h
tokenizer.o: ../../dlib/logger/logger_kernel_1.h
tokenizer.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
tokenizer.o: ../config_reader/config_reader_kernel_1.h
tokenizer.o: ../config_reader/config_reader_kernel_abstract.h
tokenizer.o: ../../dlib/map.h ../../dlib/map/map_kernel_1.h
tokenizer.o: ../../dlib/map/map_kernel_abstract.h
tokenizer.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
tokenizer.o: ../config_reader/config_reader_thread_safe_1.h
tokenizer.o: ../config_reader/config_reader_thread_safe_abstract.h
tokenizer.o: ../../dlib/assert.h ../../dlib/algs.h
tuple.o: ../../dlib/tuple.h ../../dlib/tuple/tuple.h ../enable_if.h ../algs.h
tuple.o: ../platform.h ../assert.h ../error.h ../noncopyable.h ../serialize.h
tuple.o: ../algs.h ../uintn.h ../interfaces/enumerable.h
tuple.o: ../interfaces/map_pair.h ../enable_if.h
tuple.o: ../../dlib/tuple/tuple_abstract.h tester.h ../../dlib/map.h
tuple.o: ../../dlib/logger.h ../../dlib/logger/logger_kernel_1.h ../threads.h
tuple.o: ../threads/threads_kernel.h ../platform.h ../threads/posix.h
tuple.o: ../threads/threads_kernel_2.h ../threads/threads_kernel_abstract.h
tuple.o: /usr/include/pthread.h /usr/include/features.h
tuple.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
tuple.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-32.h
tuple.o: /usr/include/sched.h /usr/include/bits/types.h
tuple.o: /usr/include/bits/typesizes.h /usr/include/time.h
tuple.o: /usr/include/bits/sched.h /usr/include/signal.h
tuple.o: /usr/include/bits/sigset.h /usr/include/bits/pthreadtypes.h
tuple.o: /usr/include/bits/setjmp.h /usr/include/errno.h
tuple.o: /usr/include/bits/errno.h /usr/include/linux/errno.h
tuple.o: /usr/include/asm/errno.h /usr/include/asm-i386/errno.h
tuple.o: /usr/include/asm-generic/errno.h
tuple.o: /usr/include/asm-generic/errno-base.h /usr/include/sys/time.h
tuple.o: /usr/include/bits/time.h /usr/include/sys/select.h
tuple.o: /usr/include/bits/select.h ../threads/threads_kernel_shared.h
tuple.o: ../threads/auto_mutex_extension.h ../threads/threads_kernel.h
tuple.o: ../threads/rmutex_extension.h ../threads/rmutex_extension_abstract.h
tuple.o: ../threads/auto_mutex_extension_abstract.h ../binary_search_tree.h
tuple.o: ../binary_search_tree/binary_search_tree_kernel_1.h
tuple.o: ../binary_search_tree/binary_search_tree_kernel_abstract.h
tuple.o: ../interfaces/map_pair.h ../interfaces/enumerable.h
tuple.o: ../interfaces/remover.h
tuple.o: ../binary_search_tree/binary_search_tree_kernel_2.h
tuple.o: ../binary_search_tree/binary_search_tree_kernel_c.h ../assert.h
tuple.o: ../../dlib/memory_manager.h ../member_function_pointer.h
tuple.o: ../member_function_pointer/member_function_pointer_kernel_1.h
tuple.o: ../member_function_pointer/member_function_pointer_kernel_abstract.h
tuple.o: ../member_function_pointer/member_function_pointer_kernel_c.h
tuple.o: ../memory_manager.h ../memory_manager/memory_manager_kernel_1.h
tuple.o: ../memory_manager/memory_manager_kernel_abstract.h
tuple.o: ../memory_manager/memory_manager_kernel_2.h
tuple.o: ../memory_manager/memory_manager_kernel_3.h
tuple.o: ../memory_manager/memory_manager_kernel_2.h
tuple.o: ../binary_search_tree/binary_search_tree_kernel_2.h ../queue.h
tuple.o: ../queue/queue_kernel_1.h ../queue/queue_kernel_abstract.h
tuple.o: ../queue/queue_kernel_2.h ../queue/queue_kernel_c.h
tuple.o: ../queue/queue_sort_1.h ../queue/queue_sort_abstract.h ../sort.h
tuple.o: ../set.h ../set/set_kernel_1.h ../set/set_kernel_abstract.h
tuple.o: ../set/set_kernel_c.h binary_search_tree.h
tuple.o: ../../dlib/memory_manager_global.h
tuple.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_1.h
tuple.o: ../memory_manager/memory_manager_kernel_abstract.h
tuple.o: ../../dlib/memory_manager_global/memory_manager_global_kernel_abstract.h
tuple.o: ../../dlib/memory_manager_stateless.h
tuple.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_1.h
tuple.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h
tuple.o: ../../dlib/memory_manager_stateless/memory_manager_stateless_kernel_2.h
tuple.o: ../../dlib/binary_search_tree.h ../set/set_compare_1.h
tuple.o: ../set/set_compare_abstract.h ../threads/auto_mutex_extension.h
tuple.o: ../threads/auto_unlock_extension.h
tuple.o: ../threads/auto_unlock_extension_abstract.h
tuple.o: ../threads/create_new_thread_extension.h
tuple.o: ../threads/create_new_thread_extension_abstract.h
tuple.o: ../threads/multithreaded_object_extension.h
tuple.o: ../threads/multithreaded_object_extension_abstract.h
tuple.o: ../threads/rsignaler_extension.h
tuple.o: ../threads/rsignaler_extension_abstract.h ../map.h
tuple.o: ../threads/rmutex_extension.h ../threads/rsignaler_extension.h
tuple.o: ../threads/threaded_object_extension.h
tuple.o: ../threads/threaded_object_extension_abstract.h
tuple.o: ../threads/thread_specific_data_extension.h
tuple.o: ../threads/thread_specific_data_extension_abstract.h
tuple.o: ../threads/thread_function_extension.h
tuple.o: ../threads/thread_function_extension_abstract.h
tuple.o: ../threads/threaded_object_extension.h ../misc_api.h
tuple.o: ../misc_api/posix.h ../misc_api/misc_api_kernel_2.h
tuple.o: ../misc_api/misc_api_kernel_abstract.h ../uintn.h
tuple.o: ../../dlib/logger/logger_kernel_abstract.h ../smart_pointers.h
tuple.o: ../smart_pointers/scoped_ptr.h ../noncopyable.h
tuple.o: ../smart_pointers/scoped_ptr_abstract.h
tuple.o: ../smart_pointers/shared_ptr.h
tuple.o: ../smart_pointers/shared_ptr_abstract.h ../smart_pointers/weak_ptr.h
tuple.o: ../smart_pointers/shared_ptr.h ../smart_pointers/weak_ptr_abstract.h
tuple.o: ../../dlib/logger/extra_logger_headers.h
tuple.o: ../../dlib/logger/logger_kernel_1.h
tuple.o: ../../dlib/logger/logger_config_file.h ../config_reader.h
tuple.o: ../config_reader/config_reader_kernel_1.h
tuple.o: ../config_reader/config_reader_kernel_abstract.h ../../dlib/map.h
tuple.o: ../../dlib/map/map_kernel_1.h ../../dlib/map/map_kernel_abstract.h
tuple.o: ../../dlib/map/map_kernel_c.h ../tokenizer.h
tuple.o: ../tokenizer/tokenizer_kernel_1.h
tuple.o: ../tokenizer/tokenizer_kernel_abstract.h
tuple.o: ../tokenizer/tokenizer_kernel_c.h
tuple.o: ../config_reader/config_reader_thread_safe_1.h
tuple.o: ../config_reader/config_reader_thread_safe_abstract.h
tuple.o: ../../dlib/assert.h ../../dlib/algs.h
