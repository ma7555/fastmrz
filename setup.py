from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()


setup(
    name="fastmrz",
    version="1.2",
    author="Sivakumar Mahalingam",
    description="Extracts the Machine Readable Zone (MRZ) data from document images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sivakumar-mahalingam/fastmrz",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
    ],
    install_requires=[
        "opencv-python>=4.9.0.80",
        "pytesseract>=0.3.10",
    ],
    python_requires=">=3.8",
)
