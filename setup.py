import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cirtorch",
    description="CNN Image Retrieval in PyTorch: "
                "Training and evaluating CNNs for Image Retrieval in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    url="https://github.com/FUAVX/IIR-Robust-Retrieval",
    packages=find_packages(),
    license="MIT",
    keywords=["machine learning", "cnn", "computer vision"],
    install_requires=[
        "numpy",
        "Pillow",
        "scipy",
        "torch",
        "torchvision",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
    ],
)
