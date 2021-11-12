from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("rpc_trans", ["rpc_trans.pyx"],
        include_dirs=["/opt/miniconda/include"],
        libraries=["gdal"],
        library_dirs=["/opt/miniconda/lib"]),
]
setup(
    ext_modules = cythonize(extensions)
)