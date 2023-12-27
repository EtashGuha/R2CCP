from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'R2CCP Package for Conformal Prediction'
LONG_DESCRIPTION = 'This package provides a class for using R2CCP (Regression to Classification for Conformal Prediction)'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="R2CCP", 
        version=VERSION,
        author="Etash Guha",
        author_email="etashguha@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        readme = "README.md",
        install_requires=[
    "ConfigArgParse>=1.7",
    "numpy>=1.22.1",
    "pyinterval>=1.2.0",
    "pytorch_lightning>=1.9.5",
    "scikit_learn>=1.3.0",
    "scipy>=1.7.3",
    "six>=1.16.0",
    'torch>=2.0.1',
    'torchvision>=0.15.2',
    'urllib3',
    "tqdm",
    "charset-normalizer",
    "idna",
    "certifi",
    "mpmath"
], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)