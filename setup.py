"""setup for the dlib project
 Copyright (C) 2015  Ehsan Azar (dashesy@linux.com)
 License: Boost Software License   See LICENSE.txt for the full license.

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


def _get_options():
    """read arguments and creates options
    """
    _cmake_path = find_executable("cmake")
    _cmake_extra = []
    _cmake_config = 'Release'

    _options = []
    opt_key = None

    # parse commandline options and consume those we care about
    for opt_idx, arg in enumerate(sys.argv):
        if opt_key == 'cmake':
            _cmake_path = arg
        elif opt_key == 'yes':
            _cmake_extra.append('-D{arg}=yes'.format(arg=arg.trim()))
        elif opt_key == 'no':
            _cmake_extra.append('-D{arg}=no'.format(arg=arg.trim()))

        if opt_key:
            sys.argv.remove(arg)
            opt_key = None
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

        opt_key = None
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

    return _options, _cmake_config, _cmake_path, _cmake_extra

options, cmake_config, cmake_path, cmake_extra = _get_options()

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
    major = readme('./docs/.current_release_number').strip()
    minor = readme('./docs/.current_minor_release_number').strip()
    return major + '.' + minor


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
            f.write('__version__ = {ver}\n'.format(ver=read_version()))
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
    long_description=readme('./README.txt'),
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
        'License :: OSI Approved :: Boost Software License (BSL)',
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
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development',
    ],
)
