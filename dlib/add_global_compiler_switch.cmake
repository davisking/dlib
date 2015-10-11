

cmake_minimum_required(VERSION 2.8.4)

# Make macros that can add compiler switches to the entire project.  Not just
# to the current cmake folder being built.  
macro ( add_global_compiler_switch switch_name )
   # If removing the switch would change the flags then it's already present
   # and we don't need to do anything.
   string(REPLACE "${switch_name}" "" tempstr "${CMAKE_CXX_FLAGS}")
   if ("${CMAKE_CXX_FLAGS}" STREQUAL "${tempstr}" )
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${switch_name}" 
         CACHE STRING "Flags used by the compiler during all C++ builds." 
         FORCE)
   endif ()
endmacro()

macro ( remove_global_compiler_switch switch_name )
   string(REPLACE "${switch_name}" "" tempstr "${CMAKE_CXX_FLAGS}")
   if (NOT "${CMAKE_CXX_FLAGS}" STREQUAL "${tempstr}" )
      set (CMAKE_CXX_FLAGS "${tempstr}" 
         CACHE STRING "Flags used by the compiler during all C++ builds." 
         FORCE)
   endif ()
endmacro()

macro (add_global_define def_name)
   add_global_compiler_switch(-D${def_name})
endmacro()

macro (remove_global_define def_name)
   remove_global_compiler_switch(-D${def_name})
endmacro()
