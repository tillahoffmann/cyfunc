from Cython.Build import cythonize
import numpy as np
from setuptools import setup, find_packages, Extension

ext = cythonize('cyfunc/*.pyx', language_level=3)


setup(
    name='cyfunc',
    version='0.1',
    install_requires=[
        'numpy',
        'cython',
    ],
    setup_requires=[
        'cython',
    ],
    extras_require={
        'tests': [
            'pytest',
        ],
    },
    package_data={
        'cyfunc': [
            '__init__.pxd',
            'cyfunc.pxd',
        ],
    },
    ext_modules=ext,
    include_dirs=[
        np.get_include(),
    ]
)
