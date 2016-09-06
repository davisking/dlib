date /T > test_log.txt
time /T >> test_log.txt

rem the pings are to wait between builds so visual studio doesn't get in a funk.



echo testing python >> test_log.txt
rm -rf build_python
mkdir build_python
cd build_python
cmake -G "Visual Studio 14 2015 Win64" ../../../tools/python  -DPYTHON3=ON
cmake --build . --config Release --target install || exit /B
ping 127.0.0.1 -n 5 -w 1000 > null
cd ..



echo testing vc2015 >> test_log.txt
rm -rf build_vc2015_64
mkdir build_vc2015_64
cd build_vc2015_64
cmake -G "Visual Studio 14 2015 Win64" .. 
cmake --build . --config Release || exit /B
ping 127.0.0.1 -n 5 -w 1000 > null
cmake --build . --config Debug || exit /B
ping 127.0.0.1 -n 5 -w 1000 > null
cd Release
dtest --runall -d || exit /B
cd ..
cd ..





del null
type test_log.txt

date /T
time /T

