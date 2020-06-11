from setuptools import setup, find_packages

setup(
    name='deepltl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow>=2.1.0',
        'py-aiger',
        'py-aiger-sat'
    ],
    python_requires='>=3.7'
)
