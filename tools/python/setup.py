"""setup for the dlib project

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
To exclude/include certain features in the build:
    --no-gui-support: sets DLIB_NO_GUI_SUPPORT
    --enable-stack-trace: sets DLIB_ENABLE_STACK_TRACE
    --enable-asserts: sets DLIB_ENABLE_ASSERTS
    --no-blas: unsets DLIB_USE_BLAS
    --no-lapack: unsets DLIB_USE_LAPACK
    --no-libpng: unsets DLIB_LINK_WITH_LIBPNG
    --no-libjpeg: unsets DLIB_LINK_WITH_LIBJPEG
    --no-sqlite3: unsets DLIB_LINK_WITH_SQLITE3
Additional options:
    --debug: makes a debug build
    --cmake: path to specific cmake executable
"""

from __future__ import print_function
import shutil
import stat
import errno

from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build import build as _build
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.spawn import find_executable
from distutils import log
import os
import sys
from setuptools import Extension, setup
import platform
from subprocess import Popen, PIPE, STDOUT
import signal
from threading import Thread
import time


# change directory to this module path
try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = os.path.abspath(this_file)
if os.path.dirname(this_file):
    os.chdir(os.path.dirname(this_file))
script_dir = os.getcwd()

cmake_path = find_executable("cmake")
cmake_extra = []
cmake_config = 'Release'
options = [arg[2:].lower() for arg in sys.argv if arg.startswith('--')]

opt_key = None
# parse commandline options
for opt_idx, opt in enumerate(options):
    if opt_key == 'cmake':
        cmake_path = opt

    if opt_key:
        continue

    if opt == 'cmake':
        cmake_path = None
        opt_key = opt
        continue

    opt_key = None
    if opt == 'debug':
        cmake_config = 'Debug'
    elif opt == 'release':
        cmake_config = 'Release'
    elif opt == 'no-gui-support':
        cmake_extra.append('-DDLIB_NO_GUI_SUPPORT=yes')
    elif opt == 'enable-stack-trace':
        cmake_extra.append('-DDLIB_ENABLE_STACK_TRACE=yes')
    elif opt == 'enable-asserts':
        cmake_extra.append('-DDLIB_ENABLE_ASSERTS=yes')
    elif opt == 'no-blas':
        cmake_extra.append('-DDLIB_USE_BLAS=no')
    elif opt == 'no-lapack':
        cmake_extra.append('-DDLIB_USE_LAPACK=no')
    elif opt == 'no-libpng':
        cmake_extra.append('-DDLIB_LINK_WITH_LIBPNG=no')
    elif opt == 'no-libjpeg':
        cmake_extra.append('-DDLIB_LINK_WITH_LIBJPEG=no')
    elif opt == 'no-sqlite3':
        cmake_extra.append('-DDLIB_LINK_WITH_SQLITE3=no')
    elif opt not in ['debug', 'release',
                     'repackage']:
        raise DistutilsOptionError("Unrecognized option {opt}".format(opt=opt))

if cmake_path is None:
    raise DistutilsSetupError("Cannot find cmake in the path. Please specify its path with --cmake parameter.")

try:
    from Queue import Queue, Empty
except ImportError:
    # noinspection PyUnresolvedReferences
    from queue import Queue, Empty  # python 3.x


_ON_POSIX = 'posix' in sys.builtin_module_names


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


def _log_buf(buf):
    if not buf:
        return
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
              close_fds=_ON_POSIX, preexec_fn=os.setsid)

    q = Queue()
    t = Thread(target=enqueue_output, args=(p.stdout, q))
    t.daemon = True  # thread dies with the program
    t.start()

    _time = time.time()
    elapsed = 0
    e = None
    try:
        while t.isAlive():
            try:
                buf = q.get(timeout=.1)
            except Empty:
                buf = ''
            _log_buf(buf)
            elapsed = time.time() - _time
            if timeout and elapsed > timeout:
                break
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
    log.info("Copying file %s to %s." % (src, dst))
    shutil.copy2(src, dst)


# noinspection PyPep8Naming
class build(_build):
    def run(self):
        repackage = 'repackage' in options
        if not repackage:
            self.build_dlib()

        dist_dir = os.path.join(script_dir, "dist")
        if os.path.exists(dist_dir):
            log.info('Removing distribution directory %s' % dist_dir)
            rmtree(dist_dir)

        dist_dir_examples = os.path.join(script_dir, "dist/dlib/examples")
        try:
            os.makedirs(dist_dir_examples)
        except OSError:
            pass

        log.info('Populating the distribution directory %s' % dist_dir)
        dist_dir = os.path.join(script_dir, "dist/dlib")

        # create the module init files
        with open(os.path.join(dist_dir, '__init__.py'), 'w'):
            pass
        with open(os.path.join(dist_dir_examples, '__init__.py'), 'w'):
            pass

        # this is where the extension and Python examples are located
        out_dir = os.path.join(script_dir, "../../python_examples")

        ext_found = False
        # manually copy everything to distribution folder with package hierarchy in mind
        names = os.listdir(out_dir)
        for name in names:
            srcname = os.path.join(out_dir, name)
            dstname = os.path.join(dist_dir, name)
            dstextname = os.path.join(dist_dir_examples, name)
            if name.endswith('.py') or name.endswith('.txt'):
                copy_file(srcname, dstextname)
            elif name.endswith('.dll') or name.endswith('.so') or name.endswith('.pyd'):
                if name.startswith('dlib'):
                    ext_found = True
                copy_file(srcname, dstname)

        if not ext_found:
            raise DistutilsSetupError("Cannot find dlib extension module.")

        return _build.run(self)

    @staticmethod
    def build_dlib():
        """use cmake to build and install the extension
        """
        platform_arch = platform.architecture()[0]
        log.info("Detected Python architecture: %s" % platform_arch)

        build_dir = os.path.join(script_dir, "build")
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
        ] + cmake_extra
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

        # cd where setup knows
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
    version='0.0.1',
    keywords=['dlib', 'Computer Vision', 'Machine Learning'],
    description='A toolkit for making real world machine learning and data analysis applications',
    long_description=readme('../../README.txt'),
    author='Davis King',
    author_email='davis@dlib.net',
    url='https://github.com/davisking/dlib',
    license='Boost Software License',
    packages=['dlib', 'dlib.examples'],
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
)
