% This example calls the three mex functions defined in this folder.  As you
% can see, you call them just like you would normal MATLAB functions.

x = magic(3)
y = 2*magic(3)

[out1, out2] = example_mex_function(x,y, 12345)

z = example_mex_callback(x, @(a)a+a)


input = {}
input.val = 2
input.stuff = 'some string'
output = example_mex_struct(input)

