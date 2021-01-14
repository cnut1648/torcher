#!/usr/bin/env python

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.readlines()

setup(
    name='Torcher',
    version='0.1.0',
    description='utils codes for pyTorch',
    author="Jiashu Xu",
    author_email='1999J0615une@gmail.com',
    url='https://github.com/cnut1648/torcher',
    download_url='https://github.com/cnut1648/torcher/archive/v0.1.tar.gz',
    packages=["Torcher"],
    license="MIT license",
    keywords=['nlp', 'skills', 'onet', 'pytorch'],
    install_requires=[            # I get to this in a second
        'beautifulsoup4',
        'numpy',
        'graphviz',
        'matplotlib',
        'requests',
        'nltk',
        'torch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)