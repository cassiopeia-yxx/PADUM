"""Setup script for the PADUM project."""
from setuptools import setup, find_packages

setup(
    name="basicsr",
    version="0.1.0",
    description="Basic SR (Super-Resolution) library for image deraining tasks",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/cassiopeia-yxx/PADUM",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-image>=0.20.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "einops>=0.6.1",
        "tb-nightly>=2.15.0a",
        "mamba-ssm>=0.0.1"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "padum-train=basicsr.train:main",
            "padum-test=basicsr.test:main"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={
        "basicsr": ["utils/matlab_functions.m"]
    }
)