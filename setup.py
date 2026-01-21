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

Cython extension build configuration for circe.pyquic
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import platform
from Cython.Build import cythonize


class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            if platform.system() == 'Darwin':
                ext.extra_compile_args = ['-I/System/Library/Frameworks/vecLib.framework/Headers']
                if 'ppc' in platform.machine():
                    ext.extra_compile_args.append('-faltivec')
                ext.extra_link_args = ['-Wl,-framework', '-Wl,Accelerate']
            else:
                ext.include_dirs.append('/usr/local/include')
                ext.extra_compile_args += ['-msse2', '-O2', '-fPIC', '-w']
                ext.extra_link_args += ['-llapack']
        build_ext.build_extensions(self)


extensions = [
    Extension(
        'circe.pyquic',
        sources=['pyquic_ext/QUIC.C', 'pyquic_ext/pyquic.pyx'],
        include_dirs=[numpy.get_include(), 'pyquic_ext'],
        libraries=['lapack', 'blas'],
        language='c++',
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    cmdclass={'build_ext': BuildExt},
)
