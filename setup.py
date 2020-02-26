import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read in requirements
requirements = [
    requirement.strip() for requirement in open('requirements.txt').readlines()
]

setuptools.setup(
    name="ghlestimator", # Replace with your own username
    version="0.0.2",
    author="Damian Draxler",
    author_email="damiandraxler01@gmail.com",
    description="Linear generalized Huber estimator compatible with scikit-learn.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/damiandraxler/ghlestimator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,        
    python_requires='>=3.7',
)