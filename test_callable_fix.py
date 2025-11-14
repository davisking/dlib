#!/usr/bin/env python3
"""Test script to verify the fix for issue #2008"""
from functools import partial

def test_function(x, y):
    """Simple test function"""
    return -(x**2 + y**2)

def test_single_arg(x):
    """Single argument function"""
    return -(x**2)

# Test with functools.partial
partial_func = partial(test_function, 2)

print("Testing functools.partial with dlib.find_max_global...")
print(f"partial_func type: {type(partial_func)}")
print(f"partial_func callable: {callable(partial_func)}")

# This should work after the fix
try:
    import dlib
    result = dlib.find_max_global(partial_func, [0.], [10.], 100)
    print(f"✓ Success! Result: {result}")
except AttributeError as e:
    print(f"✗ Failed with AttributeError: {e}")
except Exception as e:
    print(f"✗ Failed with error: {e}")

# Test with regular function (should still work)
print("\nTesting regular function with dlib.find_max_global...")
try:
    import dlib
    result = dlib.find_max_global(test_single_arg, [0.], [10.], 100)
    print(f"✓ Success! Result: {result}")
except Exception as e:
    print(f"✗ Failed with error: {e}")

# Test with lambda (should still work)
print("\nTesting lambda with dlib.find_max_global...")
try:
    import dlib
    result = dlib.find_max_global(lambda x: -(x**2), [0.], [10.], 100)
    print(f"✓ Success! Result: {result}")
except Exception as e:
    print(f"✗ Failed with error: {e}")
