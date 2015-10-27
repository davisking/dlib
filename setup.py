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
To package the wheel:
    python setup.py bdist_wheel
To repackage the previously built package as wheel (bypassing build):
    python setup.py bdist_wheel --repackage
To install a develop version (egg with symbolic link):
    python setup.py develop
To exclude/include certain options in the cmake config use --yes and --no:
    for example:
    --yes DLIB_NO_GUI_SUPPORT: will set -DDLIB_NO_GUI_SUPPORT=yes
    --no DLIB_NO_GUI_SUPPORT: will set -DDLIB_NO_GUI_SUPPORT=no
Additional options:
    --debug: makes a debug build
    --cmake: path to specific cmake executable
    --G or -G: name of a build system generator (equivalent of passing -G "name" to cmake)
"""

from __future__ import print_function
import shutil
import stat
import errno

from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build import build as _build
from distutils.errors import DistutilsSetupError
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_inc, get_python_version
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
        elif opt in ['yes', 'no']:
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


def get_msvc_win64_generator():
    """find the default MSVC generator but Win64
    This logic closely matches cmake's resolution for default build generator.
    Only we select the Win64 version of it.
    """
    try:
        import _winreg as winreg
    except ImportError:
        # noinspection PyUnresolvedReferences
        import winreg

    known_vs = {
        "6.0": "Visual Studio 6",
        "7.0": "Visual Studio 7",
        "7.1": "Visual Studio 7 .NET 2003",
        "8.0": "Visual Studio 8 2005",
        "9.0": "Visual Studio 9 2008",
        "10.0": "Visual Studio 10 2010",
        "11.0": "Visual Studio 11 2012",
        "12.0": "Visual Studio 12 2013",
        "14.0": "Visual Studio 14 2015",
    }

    newest_vs = None
    newest_ver = 0

    platform_arch = platform.architecture()[0]
    sam = winreg.KEY_WOW64_32KEY + winreg.KEY_READ if '64' in platform_arch else winreg.KEY_READ
    for vs in ['VisualStudio\\', 'VCExpress\\', 'WDExpress\\']:
        vs_key = "SOFTWARE\\Microsoft\\{vs}\\".format(vs=vs)
        try:
            root_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, vs_key, 0, sam)
        except OSError:
            continue
        try:
            sub_keys = enum_reg_key(root_key)
        except OSError:
            sub_keys = []
        winreg.CloseKey(root_key)
        if not sub_keys:
            continue

        # look to see if we have InstallDir
        for sub_key in sub_keys:
            try:
                root_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, vs_key + sub_key, 0, sam)
            except OSError:
                continue
            ins_dir = reg_value(root_key, 'InstallDir')
            winreg.CloseKey(root_key)

            if not ins_dir:
                continue

            gen_name = known_vs.get(sub_key)
            if gen_name is None:
                # if it looks like a version number
                try:
                    ver = float(sub_key)
                except ValueError:
                    continue
                gen_name = 'Visual Studio %d' % int(ver)
            else:
                ver = float(sub_key)

            if ver > newest_ver:
                newest_vs = gen_name
                newest_ver = ver

    if newest_vs:
        return ['-G', newest_vs + ' Win64']
    return []

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
    buf = buf.decode("latin-1")
    buf = buf.rstrip()
    lines = buf.splitlines()
    for line in lines:
        log.info(line)


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
            raise DistutilsSetupError("Cannot find cmake in the path. Please specify its path with --cmake parameter.")

        platform_arch = platform.architecture()[0]
        log.info("Detected Python architecture: %s" % platform_arch)

        # make sure build artifacts are generated for the version of Python currently running
        cmake_extra_arch = []
        if sys.version_info >= (3, 0):
            cmake_extra_arch += ['-DPYTHON3=yes']

        if platform_arch == '64bit' and sys.platform == "win32":
            # 64bit build on Windows

            if not generator_set:
                # see if we can deduce the 64bit default generator
                cmake_extra_arch += get_msvc_win64_generator()

            # help cmake to find Python library in 64bit Python in Windows
            #  because cmake is 32bit and cannot find PYTHON_LIBRARY from registry.
            inc_dir = get_python_inc()
            cmake_extra_arch += ['-DPYTHON_INCLUDE_DIR={inc}'.format(inc=inc_dir)]

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

setup(
    name='dlib',
    version=read_version(),
    keywords=['dlib', 'Computer Vision', 'Machine Learning'],
    description='A toolkit for making real world machine learning and data analysis applications',
    long_description=readme('README.txt'),
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
