#!/usr/bin/python

from setuptools import setup

setup(
    name="referenceqvm",
    version="0.0.0",
    author="Rigetti Computing",
    description="a reference qvm that supports protoQuil",
    packages=["referenceqvm"],
    install_requires=[
        'numpy',
        'scipy',
    ]
)
