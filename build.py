"""Build script."""
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError

from setuptools import Extension
from setuptools.command.build_ext import build_ext
import numpy
import platform

if platform.system() == "Darwin":
    extra_compile_args = ["-I/System/Library/Frameworks/vecLib.framework/Headers"]
    if "ppc" in platform.machine():
        extra_compile_args.append("-faltivec")

    extra_link_args = ["-Wl,-framework", "-Wl,Accelerate"]
    include_dirs = [numpy.get_include()]

else:
    include_dirs = [numpy.get_include(), "/usr/local/include"]
    extra_compile_args = ["-msse2", "-O2", "-fPIC", "-w"]
    extra_link_args = ["-llapack"]

extensions = [
    Extension("atacnet.pyquic",
              packages=["atacnet", "atacnet.pyquic"],
              include_dirs=include_dirs,
              sources=["atacnet/pyquic/QUIC.C", "atacnet/pyquic/pyquic.pyx"],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              language="c++"
              ),
]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            pass

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            pass


def build(setup_kwargs):
    setup_kwargs.update({"ext_modules": extensions, "cmdclass": {"build_ext": ExtBuilder}})
