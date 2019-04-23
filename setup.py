import io
import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from VERSION
__version__ = None
exec(open('VERSION').read())

short_description = 'Text pair classification toolkit.'

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'numpy >= 1.14',
    'tqdm >= 4.19.4',
    'pandas >= 0.23.1',
    'pytorch >= 1.0',
    'spacy >= 2.0'
]


setup(
    name="Lion",
    version=__version__,
    author="Lixin Su, Xinyu Ma, etc",
    author_email="",
    description=(short_description),
    license="Apache 2.0",
    keywords="text pair classification toolkit",
    url="",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 1 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
)
