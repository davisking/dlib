import os
import re
import sys
import shutil
import platform
import subprocess
from distutils import log

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


def get_extra_cmake_options():
    """read --clean, --yes, --no, and --compiler-flag options from the command line and add them as cmake switches.
    """
    _cmake_extra_options = []
    _clean_build_folder = False

    opt_key = None

    argv = [arg for arg in sys.argv]  # take a copy
    # parse command line options and consume those we care about
    for opt_idx, arg in enumerate(argv):
        if opt_key == 'compiler-flags':
            _cmake_extra_options.append('-DCMAKE_CXX_FLAGS={arg}'.format(arg=arg.strip()))
        elif opt_key == 'yes':
            _cmake_extra_options.append('-D{arg}=yes'.format(arg=arg.strip()))
        elif opt_key == 'no':
            _cmake_extra_options.append('-D{arg}=no'.format(arg=arg.strip()))

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        if arg == '--clean':
            _clean_build_folder = True
            sys.argv.remove(arg)
            continue

        if arg in ['--yes', '--no', '--compiler-flags']:
            opt_key = arg[2:].lower()
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
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        return re.search(r'version\s*([\d.]+)', out.decode()).group(1)

    def run(self):
        if platform.system() == "Windows":
            if LooseVersion(self.get_cmake_version()) < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cmake_args += cmake_extra_options 

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        build_folder = os.path.abspath(self.build_temp)

        if clean_build_folder:
            rmtree(build_folder)
        if not os.path.exists(build_folder):
            os.makedirs(build_folder)

        print("Invoking CMake: '{}'".format(['cmake', ext.sourcedir] + cmake_args))
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_folder)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_folder)


from setuptools.command.test import test as TestCommand
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

def read_version():
    """Read version information
    """
    major = re.findall("set\(CPACK_PACKAGE_VERSION_MAJOR.*\"(.*)\"", open('dlib/CMakeLists.txt').read())[0]
    minor = re.findall("set\(CPACK_PACKAGE_VERSION_MINOR.*\"(.*)\"", open('dlib/CMakeLists.txt').read())[0]
    patch = re.findall("set\(CPACK_PACKAGE_VERSION_PATCH.*\"(.*)\"", open('dlib/CMakeLists.txt').read())[0]
    return major + '.' + minor + '.' + patch

def readme(fname):
    """Read text out of a file relative to setup.py.
    """
    return open(os.path.join(fname)).read()

setup(
    name='dlib',
    version=read_version(),
    keywords=['dlib', 'Computer Vision', 'Machine Learning'],
    description='A toolkit for making real world machine learning and data analysis applications',
    long_description=readme('README.md'),
    author='Davis King',
    author_email='davis@dlib.net',
    url='https://github.com/davisking/dlib',
    license='Boost Software License',
    packages=['dlib'],
    ext_modules=[CMakeExtension('dlib','tools/python')],
    cmdclass=dict(build_ext=CMakeBuild, test=PyTest),
    zip_safe=False,
    tests_require=['pytest'],
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development',
    ],
)
