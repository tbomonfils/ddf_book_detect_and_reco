#!/usr/bin/env python3

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
#long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='fnac',
    version='0.0.1',
    description='',
    author='Thibaud LE GALL',
    url='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        '',
        '',
#        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
#    package_dir={'': 'servier'},
    packages=find_packages(),
    keywords='',
    python_requires='>=3.0, <4',
#    install_requires=['tensorflow', 'numpy', 'scikit-learn', 'pandas'],
    extras_require={
        'dev': [''],
        'test': [''],
    },
#    package_data={
#        'sample': ['dataset_multi.csv', 'dataset_single.csv'],
#    },
    entry_points={
        'console_scripts': [
            'fnac=app:main',
        ],
    },
)