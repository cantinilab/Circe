# setup.py
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy
import platform
from Cython.Build import cythonize
import sys

class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if platform.system() == "Darwin":
                ext.extra_compile_args = ["-I/System/Library/Frameworks/vecLib.framework/Headers"]
                if "ppc" in platform.machine():
                    ext.extra_compile_args.append("-faltivec")
                ext.extra_link_args = ["-Wl,-framework", "-Wl,Accelerate"]
            else:
                ext.include_dirs.append("/usr/local/include")
                ext.extra_compile_args += ["-msse2", "-O2", "-fPIC", "-w"]
                ext.extra_link_args += ["-llapack"]
        build_ext.build_extensions(self)


extensions = [
    Extension(
        "circe.pyquic",
        sources=["pyquic_ext/QUIC.C", "pyquic_ext/pyquic.pyx"],
        include_dirs=[numpy.get_include(), "pyquic_ext"],
        libraries=['lapack', 'blas'],
        language="c++",
    ),
]

setup(
    name='circe-py',
    version='0.3.4',
    description='Circe: Package for building co-accessibility networks from ATAC-seq data.',
    long_description="None",
    author='Remi-Trimbour',
    author_email='remi.trimbour@pasteur.fr',
    maintainer='Remi-Trimbour',
    maintainer_email='remi.trimbour@gmail.com',
    url='https://github.com/cantinilab/circe',
    packages=find_packages(),
    package_data={'': ['*']},
    python_requires='>=3.8,<4.0',
    ext_modules=cythonize(extensions, language_level=3),
    cmdclass={'build_ext': BuildExt},
    include_dirs=[numpy.get_include()],
    options={'bdist_wheel':{'universal':True}},
    install_requires=[
        'Cython',
        'numpy<2.0.0',
        'pandas>=2.1.1',
        'scikit-learn>=1.3.1',
        'joblib>=1.1.0',
        'scanpy>=1.8.1',
        'rich>=10.12.0',
    ],
    classifiers=[
        # Add appropriate classifiers
    ],
)
