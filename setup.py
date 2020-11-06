from setuptools import setup, find_packages, Extension


# Try to build from source if cython is installed, use preprocessed C file otherwise.
try:
    from Cython.Build import cythonize
    ext = cythonize('cyfunc/*.pyx', language_level=3)
except ImportError:
    ext = [Extension('cyfunc', ['cyfunc/__init__.c'])]


class numpy_include(object):
    """
    Lazily evaluate numpy include path because numpy may not be installed .
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


setup(
    name='cyfunc',
    version='0.1',
    packages=find_packages(),
    setup_requires=[
        'numpy',
    ],
    install_requires=[
        'numpy',
        'cython',
    ],
    extras_require={
        'tests': [
            'pytest',
            'flake8',
        ],
    },
    ext_modules=ext,
    include_dirs=[
        numpy_include(),
    ],
    package_data={
        'cyfunc': [
            '*.pxd',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
