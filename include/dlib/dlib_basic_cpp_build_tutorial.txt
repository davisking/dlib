#error "Don't write #include <dlib/all/source.cpp> in your code."
/*
  In C++, it is generally an error to #include .cpp files.  This is because it
  can lead to what are called multiply defined symbol errors.  Therefore, you
  should compile dlib/all/source.cpp into your application just like you would
  compile any other .cpp file.  
  
  If you are using Visual Studio you add .cpp files to your application using
  the solution explorer window.  Specifically, right click on Source Files,
  then select Add -> Existing Item and select the .cpp files you want to add.

  For general information on compiling dlib see http://dlib.net/compile.html
*/
