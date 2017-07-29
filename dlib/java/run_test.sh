
# build the jar and shared library of C++ code needed by the JVM
mkdir build
cd build
cmake ..
cmake --build . --config Release --target install
cd ..


# setup paths so the JVM can find our jar and shared library.
export LD_LIBRARY_PATH=.
export DYLD_LIBRARY_PATH=.
export CLASSPATH=myproject.jar:. 

# Now compile and run our java test that calls our C++ code.
javac swig_test.java
java swig_test
