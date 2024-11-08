from setuptools import setup, find_packages

setup(
    name='tgm-blackfriday',
    version='0.1.0',
    description='A package for loading and processing the Black Friday dataset.',
    author='Vojtech Jindra',
    author_email='jindravo@gmail.com',
    url='https://github.com/jindrvo1/blackfriday',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.3',
        'scikit-learn==1.5.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)