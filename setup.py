"""Setup script for tidalflow package."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(),
        package_dir={"tidalflow": "tidalflow"},
    )
