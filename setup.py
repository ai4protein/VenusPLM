from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="venusplm",  # Replace with your package name
    version="0.1.0",           # Initial version
    author="Mingcheng Li",
    author_email="lmc@mail.ecust.edu.cn",
    description="A short description of your package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4protein/VenusPLM",
    project_urls={
        "Bug Tracker": "https://github.com/ai4protein/VenusPLM/issues",
        "Documentation": "https://venusplm.readthedocs.io/",
        "Source Code": "https://github.com/ai4protein/VenusPLM",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        # See more at: https://pypi.org/classifiers/
    ],
    packages=find_packages(include=["venusplm", "venusplm.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "transformers",
        "biopython",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "venusplm=venusplm.module:function",
        ],
    },
    include_package_data=True,
    package_data={
        "venusplm": ["data/*.json", "data/*.csv"],
    },
)