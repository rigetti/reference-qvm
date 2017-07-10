#!/usr/bin/python

from setuptools import setup

setup(
    name="referenceqvm",
    version="0.1.0",
    author="Rigetti Computing",
    description="a reference qvm for simulation of pyQuil programs",
    packages=["referenceqvm"],
    install_requires=[
        'numpy',
        'scipy',
    ]
)