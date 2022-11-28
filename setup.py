from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="postmetad",
    python_requires=">3.7",
    author="Akash Pallath",
    author_email="apallath@seas.upenn.edu",
    description="Python package for post-processing PLUMED metadynamics and OPES simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apallath/postmetad",
    packages=["postmetad"],
)
