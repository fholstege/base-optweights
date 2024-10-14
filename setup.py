from setuptools import setup, find_packages


package_name = "optweights"

setup(
    name=package_name,
    version="0.1.0",
    author="Floris Holstege",
    author_email="f.g.holstege@uva.nl",
    description="A package for optimizing weights",
    packages=find_packages(where=package_name),
    package_dir={"": package_name} if package_name != "." else {},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9.12",
    ],
    python_requires=">=3.9.12"
)   