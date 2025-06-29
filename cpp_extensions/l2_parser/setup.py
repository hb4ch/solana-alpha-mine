import os
import sys
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup, Extension

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "fast_l2_parser",
        [
            "src/l2_parser.cpp",
            "python/l2_parser_pybind.cpp",
        ],
        include_dirs=[
            "include",
            pybind11.get_include(),
        ],
        language='c++',
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

# Compiler-specific optimization flags
class BuildExtOptimized(build_ext):
    """Custom build extension with optimization flags"""
    
    def build_extensions(self):
        # Compiler-specific optimizations
        if self.compiler.compiler_type == 'msvc':
            # Microsoft Visual C++
            for ext in self.extensions:
                ext.extra_compile_args = ['/O2', '/DNDEBUG']
        else:
            # GCC, Clang, etc.
            for ext in self.extensions:
                ext.extra_compile_args = [
                    '-O3',
                    '-DNDEBUG',
                    '-march=native',
                    '-mtune=native',
                    '-ffast-math',
                    '-funroll-loops',
                    '-finline-functions',
                ]
                ext.extra_link_args = ['-O3']
        
        super().build_extensions()

setup(
    name="fast_l2_parser",
    version="1.0.0",
    author="Crypto Alpha Mining Framework",
    author_email="",
    url="",
    description="High-performance L2 order book data parser",
    long_description="""
    A C++ extension for parsing L2 order book data with significant performance improvements 
    over Python-based parsing methods. Optimized for cryptocurrency trading applications.
    
    Features:
    - Ultra-fast string parsing using optimized C++ algorithms
    - Batch processing capabilities
    - Memory-efficient design
    - Robust error handling
    - Integration with NumPy and Polars
    - 10-50x performance improvement over pure Python parsing
    """,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtOptimized},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.10.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "test": ["pytest", "polars"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
