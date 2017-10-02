#!/usr/bin/python

from setuptools import setup
from referenceqvm import __version__

setup(
    name="referenceqvm",
    version=__version__,
    author="Rigetti Computing",
    author_email="softapps@rigetti.com",
    url="https://github.com/rigetticomputing/reference-qvm.git",
    description="a reference qvm for simulation of pyQuil programs",
    packages=[
        "referenceqvm"
    ],
    install_requires=[
        "numpy >= 1.11.1",
        "scipy >= 0.18.1",
        "pyquil >= 1.1.1",
        "funcsigs"
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest >= 3.0.0',
        'mock'
    ],
    license='LICENSE',
    test_suite='referenceqvm.tests'
)
