"""setup for the dlib project
 Copyright (C) 2015  Ehsan Azar (dashesy@linux.com)
 License: Boost Software License   See LICENSE.txt for the full license.

This file basically just uses CMake to compile the dlib python bindings project
located in the tools/python folder and then puts the outputs into standard
python packages.

To build dlib:
    python setup.py build
To build and install:
    python setup.py install
To upload the source distribution to PyPi
    python setup.py sdist 
    twine upload dist/dlib-*.tar.gz
To exclude certain options in the cmake config use --no:
    for example:
    --no USE_AVX_INSTRUCTIONS: will set -DUSE_AVX_INSTRUCTIONS=no
Additional options:
    --compiler-flags: pass flags onto the compiler, e.g. --compiler-flags "-Os -Wall" passes -Os -Wall onto GCC.
    -G: Set the CMake generator.  E.g. -G "Visual Studio 14 2015"
    --clean: delete any previous build folders and rebuild.  You should do this if you change any build options
             by setting --compiler-flags or --no since the last time you ran a build.  This will
             ensure the changes take effect.
    --set: set arbitrary cmake options e.g. --set CUDA_HOST_COMPILER=/usr/bin/gcc-6.4.0
           passes -DCUDA_HOST_COMPILER=/usr/bin/gcc-6.4.0 to CMake.
"""
import os
import re
import sys
import shutil
import platform
import subprocess
import multiprocessing
from distutils import log
from math import ceil,floor

from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


def get_extra_cmake_options():
    """read --clean, --no, --set, --compiler-flags, and -G options from the command line and add them as cmake switches.
    """
    _cmake_extra_options = []
    _clean_build_folder = False

    opt_key = None

    argv = [arg for arg in sys.argv]  # take a copy
    # parse command line options and consume those we care about
    for arg in argv:
        if opt_key == 'compiler-flags':
            _cmake_extra_options.append('-DCMAKE_CXX_FLAGS={arg}'.format(arg=arg.strip()))
        elif opt_key == 'G':
            _cmake_extra_options += ['-G', arg.strip()]
        elif opt_key == 'no':
            _cmake_extra_options.append('-D{arg}=no'.format(arg=arg.strip()))
        elif opt_key == 'set':
            _cmake_extra_options.append('-D{arg}'.format(arg=arg.strip()))

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        if arg == '--clean':
            _clean_build_folder = True
            sys.argv.remove(arg)
            continue

        if arg == '--yes':
            print("The --yes options to dlib's setup.py don't do anything since all these options ")
            print("are on by default.  So --yes has been removed.  Do not give it to setup.py.")
            sys.exit(1)
        if arg in ['--no', '--set', '--compiler-flags']:
            opt_key = arg[2:].lower()
            sys.argv.remove(arg)
            continue
        if arg in ['-G']:
            opt_key = arg[1:]
            sys.argv.remove(arg)
            continue

    return _cmake_extra_options, _clean_build_folder

cmake_extra_options,clean_build_folder = get_extra_cmake_options()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

def rmtree(name):
    """remove a directory and its subdirectories.
    """
    def remove_read_only(func, path, exc):
        excvalue = exc[1]
        if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            func(path)
        else:
            raise

    if os.path.exists(name):
        log.info('Removing old directory {}'.format(name))
        shutil.rmtree(name, ignore_errors=False, onerror=remove_read_only)


class CMakeBuild(build_ext):

    def get_cmake_version(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except:
            sys.stderr.write("""
================================================================================
================================================================================
================================================================================

                   CMake is not installed on your system!

    Or it is possible some broken copy of cmake is installed on your system.
    It is unfortunately very common for python package managers to include
    broken copies of cmake.  So if the error above this refers to some file
    path to a cmake file inside a python or anaconda or miniconda path then you
    should delete that broken copy of cmake from your computer.

    Instead, please get an official copy of cmake from one of these known good
    sources of an official cmake:
        - cmake.org (this is how windows users should get cmake)
        - apt install cmake (for Ubuntu or Debian based systems)
        - yum install cmake (for Redhat or CenOS based systems)

    On a linux machine you can run `which cmake` to see what cmake you are
    actually using.  If it tells you it's some cmake from any kind of python
    packager delete it and install an official cmake.

    More generally, cmake is not installed if when you open a terminal window
    and type 
       cmake --version
    you get an error.  So you can use that as a very basic test to see if you
    have cmake installed.  That is, if cmake --version doesn't run from the
    same terminal window from which you are reading this error message, then
    you have not installed cmake.  Windows users should take note that they
    need to tell the cmake installer to add cmake to their PATH.  Since you
    can't run commands that are not in your PATH.  This is how the PATH works
    on Linux as well, but failing to add cmake to the PATH is a particularly
    common problem on windows and rarely a problem on Linux.

================================================================================
================================================================================
================================================================================
""") 
            sys.exit(1)
        return re.search(r'version\s*([\d.]+)', out.decode()).group(1)

    def run(self):
        cmake_version = self.get_cmake_version()
        if platform.system() == "Windows":
            if LooseVersion(cmake_version) < '3.1.0':
                sys.stderr.write("\nERROR: CMake >= 3.1.0 is required on Windows\n\n")
                sys.exit(1)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DDLIB_USE_FFMPEG=OFF',]

        cmake_args += cmake_extra_options 

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            # Do a parallel build
            build_args += ['--', '/m'] 
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            # Do a parallel build
            build_args += ['--', '-j'+str(num_available_cpu_cores(2))]

        build_folder = os.path.abspath(self.build_temp)

        if clean_build_folder:
            rmtree(build_folder)
        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        cmake_setup = ['cmake', ext.sourcedir] + cmake_args
        cmake_build = ['cmake', '--build', '.'] + build_args

        print("Building extension for Python {}".format(sys.version.split('\n',1)[0]))
        print("Invoking CMake setup: '{}'".format(' '.join(cmake_setup)))
        sys.stdout.flush()
        subprocess.check_call(cmake_setup, cwd=build_folder)
        print("Invoking CMake build: '{}'".format(' '.join(cmake_build)))
        sys.stdout.flush()
        subprocess.check_call(cmake_build, cwd=build_folder)

def num_available_cpu_cores(ram_per_build_process_in_gb):
    if 'TRAVIS' in os.environ and os.environ['TRAVIS']=='true':
        # When building on travis-ci, just use 2 cores since travis-ci limits
        # you to that regardless of what the hardware might suggest.
        return 2
    elif 'CMAKE_BUILD_PARALLEL_LEVEL' in os.environ and os.environ['CMAKE_BUILD_PARALLEL_LEVEL'].isnumeric():
        return int(os.environ['CMAKE_BUILD_PARALLEL_LEVEL'])
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  
        mem_gib = mem_bytes/(1024.**3)
        num_cores = multiprocessing.cpu_count() 
        # make sure we have enough ram for each build process.
        mem_cores = int(floor(mem_gib/float(ram_per_build_process_in_gb)+0.5));
        # We are limited either by RAM or CPU cores.  So pick the limiting amount
        # and return that.
        return max(min(num_cores, mem_cores), 1)
    except ValueError:
        return 2 # just assume 2 if we can't get the os to tell us the right answer.


def read_version_from_cmakelists(cmake_file):
    """Read version information
    """
    major = re.findall("set\(CPACK_PACKAGE_VERSION_MAJOR.*\"(.*)\"", open(cmake_file).read())[0]
    minor = re.findall("set\(CPACK_PACKAGE_VERSION_MINOR.*\"(.*)\"", open(cmake_file).read())[0]
    patch = re.findall("set\(CPACK_PACKAGE_VERSION_PATCH.*\"(.*)\"", open(cmake_file).read())[0]
    return major + '.' + minor + '.' + patch

def read_entire_file(fname):
    """Read text out of a file relative to setup.py.  """
    return open(os.path.join(fname)).read()

setup(
    name='dlib',
    version=read_version_from_cmakelists('dlib/CMakeLists.txt'),
    description='A toolkit for making real world machine learning and data analysis applications',
    long_description='See http://dlib.net for documentation.',
    author='Davis King',
    author_email='davis@dlib.net',
    url='https://github.com/davisking/dlib',
    license='Boost Software License',
    ext_modules=[CMakeExtension('_dlib_pybind11','tools/python')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    #install_requires=['cmake'], # removed because the pip cmake package is busted, maybe someday it will be usable.
    packages=find_packages(exclude=['python_examples']),
    package_dir={'': 'tools/python'},
    keywords=['dlib', 'Computer Vision', 'Machine Learning'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development',
    ],
)
