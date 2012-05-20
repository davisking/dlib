

If you are reading this file then you must have downloaded dlib via the 
mercurial repository.  If you are new to dlib then go read the introduction
and how to compile pages at http://dlib.net/intro.html and http://dlib.net/compile.html.
If you are planning on contributing code then also read the contribution
instructions at http://dlib.net/howto_contribute.html.


COMPILING DLIB EXAMPLE PROGRAMS
   Go into the examples folder and type:
   mkdir build; cd build; cmake .. ; cmake --build .

   That will build all the examples.  Note that there is nothing to install
   when using dlib.  It's just a folder of source files.  Sometimes people
   tell me dlib should be compiled and installed as some kind of shared
   library, however, they are wrong. Please read this http://dlib.net/howto_contribute.html#9 
   before starting this argument again.  


RUNNING THE UNIT TEST SUITE
   Type the following to compile and run the dlib unit test suite (it takes a while):
   cd dlib/test; mkdir build; cd build; cmake ..; cmake --build . --config Release; ./test --runall

   Note that on windows your compiler might put the test executable in a subfolder called
   Release.  If that's the case then you have to go to that folder before running the test.


DOCUMENTATION
   The mercurial repository doesn't contain finished documentation.  The stuff in
   the docs folder is just a bunch of scripts and xml files used to generate the 
   documentation.  There is a readme in docs/README.txt which discusses how to do
   this.  However, unless you are trying to modify the documentation, you should
   just download a copy from http://dlib.net.  That would probably be easier than
   setting up your environment to generate the documentation yourself.
