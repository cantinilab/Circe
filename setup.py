"""
Circe: Predict cis-Regulatory DNA Interactions from Single-Cell ATAC-seq Data

   O - - - - - ⊙                                    O - - - - - O     
    O - - - - ⊙ *                                   *0 - - - - o      
      O - - o     **                              *    O - - o        
         O          **                           *        O           
      o - - O         **                         *     o - - O        
    o - - - - O         **                       *   o - - - - O      
   o - - - - - O       ▄▄**  ▄▄▄  ▄▄▄    ▄▄▄  ▄▄▄▄* o - - - - - O     
   O - - - - - O      █   **  █   █  █  █     █     O - - - - - O     
   O - - - - - ⊙*     █    ** █   █▀▀▄  █     █▀▀▀  O - - - - - o     
    O - - - - o  *    ▀▄▄▄ **▄█▄  █  █  ▀▄▄▄  █▄▄▄  *0 - - - - o      
      O - - o     *        **                     *    O - - o        
         O        *       **                     *        O           
      o - - O     *      **     Predict         *      o - - O        
    o - - - - O  *     **    cis-Regulatory     *    o - - - - O      
   o - - - - - ⊙*   **            DNA           *   o - - - - - O     
   O - - - - - o  **          interactions       *  O - - - - - o     
   O - - - - - ⊙*                                  *0 - - - - - O   

About:
-------
Circe is a Python package designed to build co-accessibility networks from single-cell ATAC-seq data. 
It implements the Cicero algorithm (Pliner et al., 2018) to predict cis-regulatory DNA interactions.

Author:
-------
Rémi Trimbour - remi.trimbour@gmail.com
"""

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    version='0.3.8',
    description='Circe: Package for building co-accessibility networks from ATAC-seq data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Remi-Trimbour',
    author_email='remi.trimbour@pasteur.fr',
    maintainer='Remi-Trimbour',
    maintainer_email='remi.trimbour@gmail.com',
    url='https://github.com/cantinilab/circe',
    license='GPL-3.0-only',
    license_file='LICENSE.txt',
    license_files=('LICENSE.txt',),
    packages=find_packages(),
    package_data={'': ['*']},
    python_requires='>=3.8,<4.0',
    ext_modules=cythonize(extensions, language_level=3),
    cmdclass={'build_ext': BuildExt},
    include_dirs=[numpy.get_include()],
    options={'bdist_wheel': {'universal': True}},
    install_requires=[
        'Cython',
        'numpy<2.0.0',
        'pandas>=2.1.1',
        'scikit-learn>=1.6',
        'joblib>=1.1.0',
        'scanpy>=1.8.1',
        'rich>=10.12.0',
        'dask',
        'distributed',
        # For Python < 3.12, any attrs ≥20.3 works fine
        'attrs>=20.3; python_version < "3.12"',
        # For Python ≥ 3.12 we prefer 23.2+ because it ships 3.12 wheels
        'attrs>=23.2; python_version >= "3.12"',
    ],
    extras_require={
        "downloads": ["pybiomart"],
      }  # Only if user wants gene body infos
)


# Note for the wheel:
# The wheel is not universal because of the Cython extension module.
# python -m build creates a wheel with the platform tag.
# To build the package:
# 1. python -m build
# 2. use auditwheel to make it general for manyllinux platforms
# 3. Use twine to upload the package to pypi with the tar.gz and the manylinux wheel
