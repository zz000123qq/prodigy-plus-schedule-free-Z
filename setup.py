"""Legacy shim.

All project metadata now lives in pyproject.toml (PEP 621), including the
version, which is resolved from prodigyplus/_version.py without importing the
package. This file only exists so that `python setup.py ...` style invocations
keep working.
"""

from setuptools import setup

setup()
