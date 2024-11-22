from setuptools import setup, find_packages

setup(
    name="ksa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "openai>=1.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "networkx>=3.0",
        "rdflib>=6.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.0.0",
        "streamlit>=1.25.0",
        "torch-geometric>=2.3.0",
        "wolframalpha>=5.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ]
    }
) 