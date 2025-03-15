from setuptools import setup, find_packages

setup(
    name="ips_optimizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pillow'
    ]
)