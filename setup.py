from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpha-engine",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered insider trading signal generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alpha-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "edgartools>=1.0.0",
        "yfinance>=0.2.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "python-dateutil>=2.8.0",
    ],
    entry_points={
        "console_scripts": [
            "alpha-engine=alpha_engine.main:main",
        ],
    },
)
