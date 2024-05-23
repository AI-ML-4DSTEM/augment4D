from setuptools import setup, find_packages

setup(
    name="augment4D",
    packages=find_packages(),
    description="Augmentation of 4D-STEM data for ML.",
    url="https://github.com/AI-ML-4DSTEM/augment4D",
    author="Arthur R. C. McCray",
    author_email="armccray@lbl.gov",
    license="MIT",
    install_requires=[
        'numpy >= 1.19',
        'matplotlib >= 3.4.2']
)