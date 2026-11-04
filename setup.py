import re
from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent
README = (here / "README.md").read_text(encoding="utf-8")

# Read the version without importing the package: importing it would pull in torch
# at build time.
version = re.search(
    r"""__version__\s*=\s*["']([^"']+)["']""",
    (here / "prodigyplus" / "_version.py").read_text(encoding="utf-8"),
).group(1)

setup(
  name = 'prodigy-plus-schedule-free',
  packages = find_packages(exclude=['tools*']),
  version = version,
  license='Apache-2.0',
  description = 'Automatic learning rate optimiser based on Prodigy and Schedule-Free',
  author = 'Logan Booker',
  author_email = 'me@loganbooker.dev',
  long_description=README,
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/LoganBooker/prodigy-plus-schedule-free',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'optimizers'
  ],
  install_requires=[
    'torch>=2.0'
  ],
  classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "License :: OSI Approved :: Apache Software License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
  ],
  python_requires='>=3.9',
)
