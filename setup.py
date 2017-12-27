"""setup for the dlib project
 Copyright (C) 2015  Ehsan Azar (dashesy@linux.com)
 License: Boost Software License   See LICENSE.txt for the full license.

This file basically just uses CMake to compile the dlib python bindings project
located in the tools/python folder and then puts the outputs into standard
python packages.

To build the dlib:
    python setup.py build
To build and install:
    python setup.py install
To package the wheel (after pip installing twine and wheel):
    python setup.py bdist_wheel
To upload the binary wheel to PyPi
    twine upload dist/*.whl
To upload the source distribution to PyPi
    python setup.py sdist upload
To repackage the previously built package as wheel (bypassing build):
    python setup.py bdist_wheel --repackage
To install a develop version (egg with symbolic link):
    python setup.py develop
To exclude/include certain options in the cmake config use --yes and --no:
    for example:
    --yes DLIB_NO_GUI_SUPPORT: will set -DDLIB_NO_GUI_SUPPORT=yes
    --no DLIB_NO_GUI_SUPPORT: will set -DDLIB_NO_GUI_SUPPORT=no
Additional options:
    --compiler-flags: pass flags onto the compiler, e.g. --compiler-flag "-Os -Wall" passes -Os -Wall onto GCC.
    --debug: makes a debug build
    --cmake: path to specific cmake executable
    --G or -G: name of a build system generator (equivalent of passing -G "name" to cmake)
"""

from __future__ import print_function
import shutil
import stat
import errno

import subprocess
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build import build as _build
from distutils.errors import DistutilsSetupError
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_inc, get_python_version, get_config_var
from distutils.version import LooseVersion
from distutils import log
import os
import sys
from setuptools import Extension, setup
import platform
from subprocess import Popen, PIPE, STDOUT
import signal
from threading import Thread
import time
import re
import pkg_resources
import textwrap


# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if os.path.dirname(this_file):
    os.chdir(os.path.dirname(this_file))
script_dir = os.getcwd()


def _get_options():
    """read arguments and creates options
    """
    _cmake_path = find_executable("cmake")
    _cmake_extra = []
    _cmake_config = 'Release'

    _options = []
    opt_key = None
    _generator_set = False  # if a build generator is set

    argv = [arg for arg in sys.argv]  # take a copy
    # parse commandline options and consume those we care about
    for opt_idx, arg in enumerate(argv):
        if opt_key == 'cmake':
            _cmake_path = arg
        elif opt_key == 'compiler-flags':
            _cmake_extra.append('-DCMAKE_CXX_FLAGS={arg}'.format(arg=arg.strip()))
        elif opt_key == 'yes':
            _cmake_extra.append('-D{arg}=yes'.format(arg=arg.strip()))
        elif opt_key == 'no':
            _cmake_extra.append('-D{arg}=no'.format(arg=arg.strip()))
        elif opt_key == 'G':
            _cmake_extra += ['-G', arg.strip()]
            _generator_set = True

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
            continue

        # Keep -G to resemble cmake's
        if arg == '-G' or arg.lower() == '--g':
            opt_key = 'G'
            sys.argv.remove(arg)
            continue

        if not arg.startswith('--'):
            continue

        opt = arg[2:].lower()
        if opt == 'cmake':
            _cmake_path = None
            opt_key = opt
            sys.argv.remove(arg)
            continue
        elif opt in ['yes', 'no', 'compiler-flags']:
            opt_key = opt
            sys.argv.remove(arg)
            continue

        custom_arg = True
        if opt == 'debug':
            _cmake_config = 'Debug'
        elif opt == 'release':
            _cmake_config = 'Release'
        elif opt in ['repackage']:
            _options.append(opt)
        else:
            custom_arg = False
        if custom_arg:
            sys.argv.remove(arg)

    return _options, _cmake_config, _cmake_path, _cmake_extra, _generator_set

options, cmake_config, cmake_path, cmake_extra, generator_set = _get_options()


def reg_value(rk, rname):
    """enumerate the subkeys in a registry key
    :param rk: root key in registry
    :param rname: name of the value we are interested in
    """
    try:
        import _winreg as winreg
    except ImportError:
        # noinspection PyUnresolvedReferences
        import winreg

    count = 0
    try:
        while True:
            name, value, _ = winreg.EnumValue(rk, count)
            if rname == name:
                return value
            count += 1
    except OSError:
        pass

    return None


def enum_reg_key(rk):
    """enumerate the subkeys in a registry key
    :param rk: root key in registry
    """
    try:
        import _winreg as winreg
    except ImportError:
        # noinspection PyUnresolvedReferences
        import winreg

    sub_keys = []
    count = 0
    try:
        while True:
            name = winreg.EnumKey(rk, count)
            sub_keys.append(name)
            count += 1
    except OSError:
        pass

    return sub_keys



try:
    from Queue import Queue, Empty
except ImportError:
    # noinspection PyUnresolvedReferences
    from queue import Queue, Empty  # python 3.x


_ON_POSIX = 'posix' in sys.builtin_module_names


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)


def _log_buf(buf):
    if not buf:
        return
    if sys.stdout.encoding:
        buf = buf.decode(sys.stdout.encoding)
    buf = buf.rstrip()
    lines = buf.splitlines()
    for line in lines:
        log.info(line)

def get_cmake_version(cmake_path):
    p = re.compile("version ([0-9.]+)")
    cmake_output = subprocess.check_output(cmake_path + " --version").decode("utf-8");
    return p.search(cmake_output).group(1)

def run_process(cmds, timeout=None):
    """run a process asynchronously
    :param cmds: list of commands to invoke on a shell e.g. ['make', 'install']
    :param timeout: timeout in seconds (optional)
    """

    # open process as its own session, and with no stdout buffering
    p = Popen(cmds,
              stdout=PIPE, stderr=STDOUT,
              bufsize=1,
              close_fds=_ON_POSIX, preexec_fn=os.setsid if _ON_POSIX else None)

    q = Queue()
    t = Thread(target=enqueue_output, args=(p.stdout, q))
    t.daemon = True  # thread dies with the program
    t.start()

    _time = time.time()
    e = None
    try:
        while t.isAlive():
            try:
                buf = q.get(timeout=.1)
            except Empty:
                buf = b''
            _log_buf(buf)
            elapsed = time.time() - _time
            if timeout and elapsed > timeout:
                break
        # Make sure we print all the output from the process.
        if p.stdout:
            for line in p.stdout:
                _log_buf(line)
            p.wait()
    except (KeyboardInterrupt, SystemExit) as e:
        # if user interrupted
        pass

    # noinspection PyBroadException
    try:
        os.kill(p.pid, signal.SIGINT)
    except (KeyboardInterrupt, SystemExit) as e:
        pass
    except:
        pass

    # noinspection PyBroadException
    try:
        if e:
            os.kill(p.pid, signal.SIGKILL)
        else:
            p.wait()
    except (KeyboardInterrupt, SystemExit) as e:
        # noinspection PyBroadException
        try:
            os.kill(p.pid, signal.SIGKILL)
        except:
            pass
    except:
        pass

    t.join(timeout=0.1)
    if e:
        raise e

    return p.returncode


def readme(fname):
    """Read text out of a file relative to setup.py.
    """
    return open(os.path.join(script_dir, fname)).read()


def read_version():
    """Read version information
    """
    major = re.findall("set\(CPACK_PACKAGE_VERSION_MAJOR.*\"(.*)\"", open('dlib/CMakeLists.txt').read())[0]
    minor = re.findall("set\(CPACK_PACKAGE_VERSION_MINOR.*\"(.*)\"", open('dlib/CMakeLists.txt').read())[0]
    patch = re.findall("set\(CPACK_PACKAGE_VERSION_PATCH.*\"(.*)\"", open('dlib/CMakeLists.txt').read())[0]
    return major + '.' + minor + '.' + patch


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
    shutil.rmtree(name, ignore_errors=False, onerror=remove_read_only)


def copy_file(src, dst):
    """copy a single file and log
    """
    log.info("Copying file %s -> %s." % (src, dst))
    shutil.copy2(src, dst)


def clean_dist():
    """re-create the dist folder
    """
    dist_dir = os.path.join(script_dir, "./dist")
    if os.path.exists(dist_dir):
        log.info('Removing distribution directory %s' % dist_dir)
        rmtree(dist_dir)

    dist_dir = os.path.join(script_dir, "./dist/dlib")
    try:
        os.makedirs(dist_dir)
    except OSError:
        pass


# always start with a clean slate
clean_dist()


# noinspection PyPep8Naming
class build(_build):
    def run(self):
        repackage = 'repackage' in options
        if not repackage:
            self.build_dlib()

        # this is where the extension examples go
        dist_dir_examples = os.path.join(script_dir, "./dist/dlib/examples")
        try:
            os.makedirs(dist_dir_examples)
        except OSError:
            pass

        # this is where the extension goes
        dist_dir = os.path.join(script_dir, "./dist/dlib")
        log.info('Populating the distribution directory %s ...' % dist_dir)

        # create the module init files
        with open(os.path.join(dist_dir, '__init__.py'), 'w') as f:
            # just so that we can `import dlib` and not `from dlib import dlib`
            f.write('from .dlib import *\n')
            # add version here
            f.write('__version__ = "{ver}"\n'.format(ver=read_version()))
        with open(os.path.join(dist_dir_examples, '__init__.py'), 'w'):
            pass

        # this is where the extension and Python examples are located
        out_dir = os.path.join(script_dir, "./python_examples")

        # these are the created artifacts we want to package
        dll_ext = ['.so']
        if sys.platform == "win32":
            dll_ext = ['.pyd', '.dll']

        ext_found = False
        # manually copy everything to distribution folder with package hierarchy in mind
        names = os.listdir(out_dir)
        for name in names:
            srcname = os.path.join(out_dir, name)
            dstname = os.path.join(dist_dir, name)
            dstextname = os.path.join(dist_dir_examples, name)

            name, extension = os.path.splitext(name.lower())
            if extension in ['.py', '.txt']:
                copy_file(srcname, dstextname)
            elif extension in dll_ext:
                if name.startswith('dlib'):
                    ext_found = True
                copy_file(srcname, dstname)

        if not ext_found:
            raise DistutilsSetupError("Cannot find built dlib extension module.")

        return _build.run(self)

    @staticmethod
    def build_dlib():
        """use cmake to build and install the extension
        """
        if cmake_path is None:
            cmake_install_url = "https://cmake.org/install/"
            message = ("You can install cmake using the instructions at " +
                       cmake_install_url)
            msg_pkgmanager = ("You can install cmake on {0} using "
                              "`sudo {1} install cmake`.")
            if sys.platform == "darwin":
                pkgmanagers = ('brew', 'port')
                for manager in pkgmanagers:
                    if find_executable(manager) is not None:
                        message = msg_pkgmanager.format('OSX', manager)
                        break
            elif sys.platform.startswith('linux'):
                try:
                    import distro
                except ImportError as err:
                    import pip
                    pip_exit = pip.main(['install', '-q', 'distro'])
                    if pip_exit > 0:
                        log.debug("Unable to install `distro` to identify "
                                  "the recommended command. Falling back "
                                  "to default error message.")
                        distro = err
                    else:
                        import distro
                if not isinstance(distro, ImportError):
                    distname = distro.id()
                    if distname in ('debian', 'ubuntu'):
                        message = msg_pkgmanager.format(
                            distname.title(), 'apt-get')
                    elif distname in ('fedora', 'centos', 'redhat'):
                        pkgmanagers = ("dnf", "yum")
                        for manager in pkgmanagers:
                            if find_executable(manager) is not None:
                                message = msg_pkgmanager.format(
                                    distname.title(), manager)
                                break
            raise DistutilsSetupError(
                "Cannot find cmake, ensure it is installed and in the path.\n"
                + message + "\n"
                "You can also specify its path with --cmake parameter.")

        platform_arch = platform.architecture()[0]
        log.info("Detected Python architecture: %s" % platform_arch)

        # make sure build artifacts are generated for the version of Python currently running
        cmake_extra_arch = []

        inc_dir = get_python_inc()
        lib_dir = get_config_var('LIBDIR')
        if (inc_dir != None):
            cmake_extra_arch += ['-DPYTHON_INCLUDE_DIR=' + inc_dir]
        if (lib_dir != None):
            cmake_extra_arch += ['-DCMAKE_LIBRARY_PATH=' + lib_dir]

        if sys.version_info >= (3, 0):
            cmake_extra_arch += ['-DPYTHON3=yes']

        log.info("Detected platform: %s" % sys.platform)
        if sys.platform == "darwin":
            # build on OS X

            # by default, cmake will choose the system python lib in /usr/lib
            # this checks the sysconfig and will correctly pick up a brewed python lib
            # e.g. in /usr/local/Cellar
            py_ver = get_python_version()
            # check: in some virtual environments the libpython has the form "libpython_#m.dylib
            py_lib = os.path.join(get_config_var('LIBDIR'), 'libpython'+py_ver+'.dylib')
            if not os.path.isfile(py_lib):
                py_lib = os.path.join(get_config_var('LIBDIR'), 'libpython'+py_ver+'m.dylib')
                
            cmake_extra_arch += ['-DPYTHON_LIBRARY={lib}'.format(lib=py_lib)]

        if sys.platform == "win32":
            if platform_arch == '64bit':
                cmake_extra_arch += ['-DCMAKE_GENERATOR_PLATFORM=x64']
                # Setting the cmake generator only works in versions of cmake >= 3.1
                if (LooseVersion(get_cmake_version(cmake_path)) < LooseVersion("3.1.0")):
                    raise DistutilsSetupError(
                        "You need to install a newer version of cmake. Version 3.1 or newer is required.");

            # this imitates cmake in path resolution
            py_ver = get_python_version()
            for ext in [py_ver.replace(".", "") + '.lib', py_ver + 'mu.lib', py_ver + 'm.lib', py_ver + 'u.lib']:
                py_lib = os.path.abspath(os.path.join(inc_dir, '../libs/', 'python' + ext))
                if os.path.exists(py_lib):
                    cmake_extra_arch += ['-DPYTHON_LIBRARY={lib}'.format(lib=py_lib)]
                    break

        build_dir = os.path.join(script_dir, "./tools/python/build")
        if os.path.exists(build_dir):
            log.info('Removing build directory %s' % build_dir)
            rmtree(build_dir)

        try:
            os.makedirs(build_dir)
        except OSError:
            pass

        # cd build
        os.chdir(build_dir)
        log.info('Configuring cmake ...')
        cmake_cmd = [
            cmake_path,
            "..",
        ] + cmake_extra + cmake_extra_arch
        if run_process(cmake_cmd):
            raise DistutilsSetupError("cmake configuration failed!")

        log.info('Build using cmake ...')

        cmake_cmd = [
            cmake_path,
            "--build", ".",
            "--config", cmake_config,
            "--target", "install",
        ]

        if run_process(cmake_cmd):
            raise DistutilsSetupError("cmake build failed!")

        # cd back where setup awaits
        os.chdir(script_dir)


# noinspection PyPep8Naming
class develop(_develop):

    def __init__(self, *args, **kwargs):
        _develop.__init__(self, *args, **kwargs)

    def run(self):
        self.run_command("build")
        return _develop.run(self)


# noinspection PyPep8Naming
class bdist_egg(_bdist_egg):
    def __init__(self, *args, **kwargs):
        _bdist_egg.__init__(self, *args, **kwargs)

    def run(self):
        self.run_command("build")
        return _bdist_egg.run(self)


# noinspection PyPep8Naming
class build_ext(_build_ext):
    def __init__(self, *args, **kwargs):
        _build_ext.__init__(self, *args, **kwargs)

    def run(self):
        # cmake will do the heavy lifting, just pick up the fruits of its labour
        pass

def is_installed(requirement):
    try:
        pkg_resources.require(requirement)
    except pkg_resources.ResolutionError:
        return False
    else:
        return True

if not is_installed('numpy>=1.5.1'):
    print(textwrap.dedent("""
            Warning: Functions that return numpy arrays need Numpy (>= v1.5.1) installed!
            You can install numpy and then run this setup again:
            $ pip install numpy
            """), file=sys.stderr)

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
    package_dir={'': 'dist'},
    include_package_data=True,
    cmdclass={
        'build': build,
        'build_ext': build_ext,
        'bdist_egg': bdist_egg,
        'develop': develop,
    },
    zip_safe=False,
    ext_modules=[Extension('dlib', [])],
    ext_package='dlib',
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
