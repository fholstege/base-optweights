from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="optweights",
    version="0.1.0",
    author="Floris Holstege",
    author_email="f.g.holstege@uva.nl",
    description="A package for optimizing weights",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11"
)