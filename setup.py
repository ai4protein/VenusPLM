from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
# Read LICENSE file
with open("LICENSE", "r", encoding="utf-8") as fh:
    license = fh.read()

setup(
    name="vplm",  # Replace with your package name
    version="0.0.1",           # Initial version
    author="Mingcheng Li",
    author_email="lmc@mail.ecust.edu.cn",
    description="VenusPLM is a protein language model that designed for protein representation and protein design.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai4protein/VenusPLM",
    project_urls={
        "Bug Tracker": "https://github.com/ai4protein/VenusPLM/issues",
        "Documentation": "https://venusplm.readthedocs.io/",
        "Source Code": "https://github.com/ai4protein/VenusPLM",
    },
    license=license,
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
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "transformers",
        "protobuf"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "vplm": ["models/vplm/vocab.txt"]
    }
)