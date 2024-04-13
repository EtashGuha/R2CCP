from setuptools import setup, find_packages
from pathlib import Path
current_directory = Path(__file__).parent
VERSION = '0.0.8' 
DESCRIPTION = 'R2CCP Package for Conformal Prediction'
LONG_DESCRIPTION = (current_directory / "README.md").read_text()

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="R2CCP", 
        version=VERSION,
        author="Etash Guha",
        author_email="etashguha@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        packages=find_packages(),
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