import setuptools

from ckb.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="ckb",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="ckb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/ckb",
    packages=setuptools.find_packages(),
    install_requires=required + ['mkb>=0.0.1'],
    package_data={
        'ckb': [
            'datasets/semanlink/*.csv'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],

    python_requires='>=3.6',
)
