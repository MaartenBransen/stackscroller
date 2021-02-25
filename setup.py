from setuptools import setup, find_packages

setup(
    name="stackscroller",
    version="0.1.0",
    author="Maarten Bransen",
    author_email="m.bransen@uu.nl",
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    packages=find_packages(include=["stackscroller"]),
    install_requires=[
        "numpy>=1.19.2",
        "matplotlib>=3.0.0",
        "pandas>=1.2.1",
    ],
)
