

# Make macros that can add compiler switches to the entire project.  Not just
# to the current cmake folder being built.  
macro ( add_global_compiler_switch switch_name )
   if (NOT CMAKE_CXX_FLAGS MATCHES "${switch_name}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${switch_name}" 
         CACHE STRING "Flags used by the compiler during all C++ builds." 
         FORCE)
   endif ()
endmacro()

macro ( remove_global_compiler_switch switch_name )
   if (CMAKE_CXX_FLAGS MATCHES " ${switch_name}")
      string (REGEX REPLACE " ${switch_name}" "" temp_var ${CMAKE_CXX_FLAGS}) 
      set (CMAKE_CXX_FLAGS "${temp_var}" 
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
