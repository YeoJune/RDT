"""Setup script for RDT package"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        return f.read()

setup(
    name='rdt-transformer',
    version='0.1.0',
    author='RDT Contributors',
    author_email='',
    description='Recursive Denoising Transformer - Progressive text denoising with adaptive computation',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/rdt',
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'datasets>=2.14.0',
        'transformers>=4.30.0',
        'pyyaml>=6.0',
        'tensorboard>=2.13.0',
        'tqdm>=4.65.0',
        'numpy>=1.24.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'rdt-train=rdt.scripts.train:main',
            'rdt-inference=rdt.scripts.inference:main',
        ],
    },
    include_package_data=True,
    package_data={
        'rdt': ['configs/*.yaml'],
    },
    zip_safe=False,
)
