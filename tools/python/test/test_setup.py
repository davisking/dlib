import pathlib
import platform
import runpy
import subprocess
import sys
import sysconfig

import pytest
import setuptools


@pytest.mark.parametrize(
    "python_platform, host_machine, expected_arch",
    [
        ("win-arm64", "ARM64", "ARM64"),
        ("win-amd64", "ARM64", "x64"),
        ("win-amd64", "AMD64", "x64"),
    ],
)
def test_windows_extension_matches_python_architecture(
    monkeypatch, tmp_path, python_platform, host_machine, expected_arch
):
    root = pathlib.Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["setup.py"])
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    setup = runpy.run_path(str(root / "setup.py"))

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: host_machine)
    monkeypatch.setattr(sysconfig, "get_platform", lambda: python_platform)
    monkeypatch.setattr(sys, "maxsize", 2**63 - 1)

    commands = []
    monkeypatch.setattr(
        subprocess, "check_call", lambda args, **kwargs: commands.append(args)
    )
    build = setup["CMakeBuild"](setuptools.Distribution())
    build.initialize_options()
    build.build_temp = str(tmp_path / "build")
    build.build_lib = str(tmp_path / "lib")
    build.build_extension(setup["CMakeExtension"]("_dlib_pybind11", "tools/python"))

    configure = commands[0]
    assert configure[configure.index("-A") + 1] == expected_arch
    assert "-DPython_EXECUTABLE=" + sys.executable in configure
