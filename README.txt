

See http://dlib.net for the main project documentation.



COMPILING DLIB EXAMPLE PROGRAMS
   Go into the examples folder and type:
   mkdir build; cd build; cmake .. ; cmake --build .

   That will build all the examples.  There is nothing to install when using
   dlib.  It's just a folder of source files.  

RUNNING THE UNIT TEST SUITE
   Type the following to compile and run the dlib unit test suite:
       cd dlib/test
       mkdir build
       cd build
       cmake ..
       cmake --build . --config Release
       ./dtest --runall

   Note that on windows your compiler might put the test executable in a
   subfolder called Release.  If that's the case then you have to go to that
   folder before running the test.

DOCUMENTATION
   The source control repository doesn't contain finished documentation.  The
   stuff in the docs folder is just a bunch of scripts and xml files used to
   generate the documentation.  There is a readme in docs/README.txt which
   discusses how to do this.  However, unless you are trying to modify the
   documentation, you should just download a copy from http://dlib.net.  
