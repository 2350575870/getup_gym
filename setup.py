from setuptools import find_packages, setup

setup(
    name="getup_gym",
    version="1.0.0",
    author="Haidong Hou",
    author_email="houhaidong@bit.edu.cn",
    maintainer="Haidong Hou",
    description="Force-Guided Fall Recovery for Bipedal-Wheeled and Humanoid Robots",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "tensorboard>=1.15",
        "onnx",
        "isaacgym-stubs",
        "pandas",
        "scikit-learn",
        "tqdm",
        "torch",
        "numpy",
    ],
    python_requires=">=3.8",
)
