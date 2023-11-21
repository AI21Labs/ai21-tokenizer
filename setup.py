from setuptools import setup, find_packages
from version import __version__
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\\n" + fh.read()

setup(
    name="ai21_tokenizer",
    version=__version__,
    author="AI21 Labs",
    author_email="support@ai21.com",
    description="Library for tokenizing text using AI21 Labs' Jurassic ai21_tokenizer",
    url="https://github.com/AI21Labs/ai21_tokenizer",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(exclude=["tests", "tests.*", "ci_scripts"]),
    install_requires=[],
    keywords=["python", "ai21_tokenizer", "ai", "ai21", "jurassic"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
