from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from numpy import get_include
from os import path
from glob import glob

def get_numpy_include_dirs():
    return [get_include()]

class build_ext_openmp(build_ext):
    openmp_compile_args = {
        'msvc': [['/openmp']],
        'intel': [['-qopenmp']],
        '*': [['-fopenmp'], ['-Xpreprocessor', '-fopenmp']],
    }
    openmp_link_args = openmp_compile_args

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith('intel'):
            compiler = 'intel'
        if compiler not in self.openmp_compile_args:
            compiler = '*'

        compile_original = self.compiler._compile

        def compile_patched(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.lower().endswith('.c'):
                extra_postargs = [arg for arg in extra_postargs if not arg.lower().startswith('-std')]
            return compile_original(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = compile_patched

        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args = list(ext.extra_link_args)

        for compile_args, link_args in zip(self.openmp_compile_args[compiler], self.openmp_link_args[compiler]):
            try:
                ext.extra_compile_args = _extra_compile_args + compile_args
                ext.extra_link_args = _extra_link_args + link_args
                return super(build_ext_openmp, self).build_extension(ext)
            except:
                print(f">>> compiling with '{' '.join(compile_args)}' failed")

        print('>>> compiling with OpenMP support failed, re-trying without')
        ext.extra_compile_args = _extra_compile_args
        ext.extra_link_args = _extra_link_args
        return super(build_ext_openmp, self).build_extension(ext)

_dir = ''

with open(path.join(_dir, 'README.md'), encoding="utf-8") as f:
    long_description = f.read()

external_root = path.join(_dir, 'stardist', 'lib', 'external')

qhull_root = path.join(external_root, 'qhull_src', 'src')
qhull_src = sorted(glob(path.join(qhull_root, '*', '*.c*')))[::-1]

nanoflann_root = path.join(external_root, 'nanoflann')

clipper_root = path.join(external_root, 'clipper')
clipper_src = sorted(glob(path.join(clipper_root, '*.cpp*')))[::-1]

setup(
    name='stardist',
    version='1.0.0',  # Set the version number directly here
    description='StarDist - Object Detection with Star-convex Shapes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stardist/stardist',
    author='Uwe Schmidt, Martin Weigert',
    author_email='research@uweschmidt.org, martin.weigert@epfl.ch',
    license='BSD-3-Clause',
    packages=find_packages(),
    python_requires='>=3.6',

    cmdclass={'build_ext': build_ext_openmp},

    ext_modules=[
        Extension(
            'stardist.lib.stardist2d',
            sources=['stardist/lib/stardist2d.cpp', 'stardist/lib/utils.cpp'] + clipper_src,
            extra_compile_args=['-std=c++11'],
            include_dirs=get_numpy_include_dirs() + [clipper_root, nanoflann_root],
        ),
        Extension(
            'stardist.lib.stardist3d',
            sources=['stardist/lib/stardist3d.cpp', 'stardist/lib/stardist3d_impl.cpp', 'stardist/lib/utils.cpp'] + qhull_src,
            extra_compile_args=['-std=c++11'],
            include_dirs=get_numpy_include_dirs() + [qhull_root, nanoflann_root],
        ),
        Extension(
            'stardist.lib.starfinity',
            sources=['stardist/lib/starfinity.cpp'],
            include_dirs=get_numpy_include_dirs(),
        )
    ],

    package_data={'stardist': ['kernels/*.cl', 'data/images/*']},

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',

        'Operating System :: OS Independent',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    install_requires=[
        'csbdeep>=0.8.0',
        'scikit-image',
        'numba',
        'imageio',
        'dask'
    ],

    extras_require={
        "tf1": ["csbdeep[tf1]>=0.8.0"],
        "test": [
            "pytest; python_version< '3.7'",
            "pytest>=7.2.0; python_version>='3.7'",
        ],
        "bioimageio": ["bioimageio.core>=0.5.0", "importlib-metadata"],
    },

    entry_points={
        'console_scripts': [
            'stardist-predict2d = stardist.scripts.predict2d:main',
            'stardist-predict3d = stardist.scripts.predict3d:main',
        ],
    }

)
