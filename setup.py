from setuptools import setup, find_packages

import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="hawk",
    version="0.1",
    description="ML metrics for Prometheus",
    long_description=README,
    long_description_content_type="text/markdown",
    author="shreyashankar",
    author_email="shreya@cs.stanford.edu",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "prometheus-client",
        "grafanalib",
        "pandas",
        "scikit-learn",
    ],
)
