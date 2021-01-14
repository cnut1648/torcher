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
    packages=["Torcher"],
    license="MIT license",
    keywords='nlp jobs skills onet pytorch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)