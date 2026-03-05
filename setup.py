from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="tinymind",
    version="0.1.0",
    description="End-to-end LLM distillation with CoT injection and mobile deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TinyMind Contributors",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
        ],
        "export": [
            "onnx>=1.15",
            "onnxruntime>=1.17",
            "coremltools>=7.0; sys_platform=='darwin'",
        ],
    },
    entry_points={
        "console_scripts": [
            "tinymind-train=tinymind.cli:train",
            "tinymind-export=tinymind.cli:export",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
