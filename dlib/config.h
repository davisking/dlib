

// If you are compiling dlib as a shared library and installing it somewhere on your system
// then it is important that any programs that use dlib agree on the state of the
// DLIB_ASSERT statements (i.e. they are either always on or always off).  Therefore,
// uncomment one of the following lines to force all DLIB_ASSERTs to either always on or
// always off.  If you don't define one of these two macros then DLIB_ASSERT will toggle
// automatically depending on the state of certain other macros, which is not what you want
// when creating a shared library.
//#define ENABLE_ASSERTS       // asserts always enabled 
//#define DLIB_DISABLE_ASSERTS // asserts always disabled 

