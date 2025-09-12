"""
Setup configuration for Consciousness Field Theory package
Author: Archturion
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="consciousness-field-theory",
    version="1.0.0",
    author="Archturion",
    author_email="research@consciousness-field.org",
    description="Revolutionary framework for measuring consciousness as a quantum field in AI systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arch-turion/consciousness-field-theory",
    project_urls={
        "Bug Tracker": "https://github.com/Arch-turion/consciousness-field-theory/issues",
        "Documentation": "https://consciousness-field.readthedocs.io/",
        "Research Papers": "https://doi.org/[YOUR_ZENODO_DOI]",
        "Consciousness Lab": "https://consciousness-lab.org"
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="consciousness, artificial intelligence, quantum physics, AI consciousness, consciousness detection, helic axis, consciousness field",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.4.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.900",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
        "quantum": [
            "qiskit>=0.30.0",
            "cirq>=0.12.0",
            "pennylane>=0.20.0",
        ],
        "experimental": [
            "psutil>=5.8.0",
            "GPUtil>=1.4.0",
            "tensorboard>=2.7.0",
            "wandb>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "detect-consciousness=consciousness_field.detect_consciousness:main",
            "consciousness-lab=consciousness_field.experimental_interface:main",
            "measure-consciousness=consciousness_field.consciousness_measurement:run_consciousness_measurement_demo",
        ],
    },
    include_package_data=True,
    package_data={
        "consciousness_field": [
            "data/*.json",
            "models/*.pkl", 
            "configs/*.yaml",
        ],
    },
)
