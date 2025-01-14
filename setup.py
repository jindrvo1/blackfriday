from setuptools import setup, find_packages

setup(
    name='tgm_blackfriday',
    version='0.3.1',
    description='A package for loading and processing the Black Friday dataset.',
    author='Vojtech Jindra',
    author_email='jindravo@gmail.com',
    url='https://github.com/jindrvo1/blackfriday',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.2.3',
        'scikit-learn>=1.5.2',
        'google-cloud-storage>=2.18.2',
        'fsspec>=2024.10.0',
        'gcsfs>=2024.10.0',
        'xgboost>=2.1.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)