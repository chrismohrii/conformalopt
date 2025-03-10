from setuptools import setup, find_packages

setup(
    name="onlinecp",  # Replace with your package name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",  # For numerical computations
        "matplotlib",  # For plotting
        "tqdm",  # For progress bars
        "statsmodels",  # For statistical models
        "cvxpy",  # For convex optimization
    ],
    extras_require={
        "dev": [  # Dependencies needed for development
            "sphinx",  # Documentation tool
            "pytest",  # Testing tool
        ],
    },
    author="Christopher Mohri",
    author_email="xmohri@stanford.edu",
    description="Simple implementation of online conformal algorithms.",
    # url="https://github.com/yourusername/my_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
