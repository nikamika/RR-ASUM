from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


'''
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("_JST",["_JST.pyx"], include_dirs=[np.get_include()])]
)
'''

extensions = [
    Extension("_JST_RR", ["_JST_RR.pyx"],
        include_dirs=[np.get_include()]
              )
]
setup(
    name="_JST_RR",
    ext_modules=cythonize(extensions)
)
