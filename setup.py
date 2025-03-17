from setuptools import setup, find_packages, find_namespace_packages

setup(
    name="ips_optimizer",
    version="0.1",
    packages=find_namespace_packages(include=['test*']),
    package_dir={'': '.'},
    install_requires=[
        'torch',
        'torchvision',
        'pillow'
    ]
)