from setuptools import setup, find_packages, Extension


# Try to build from source if cython is installed, use preprocessed C file otherwise.
try:
    from Cython.Build import cythonize
    ext = cythonize('cyfunc/*.pyx', language_level=3)
except ImportError:
    ext = [Extension('cyfunc', ['cyfunc/__init__.c'])]


class numpy_include(object):
    """
    Lazily evaluate numpy include path because numpy may not be installed.
    """
    def __str__(self):
        import numpy
        return numpy.get_include()


with open('README.rst') as fp:
    long_description = fp.read()


setup(
    name='cyfunc',
    version='0.1.2',
    packages=find_packages(),
    long_description_content_type="text/x-rst",
    long_description=long_description,
    author='Till Hoffmann',
    url='https://github.com/tillahoffmann/cyfunc',
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
            '*.pyx',
        ],
    },
    zip_safe=False,
)
