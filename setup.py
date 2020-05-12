from setuptools import setup

setup(

    name='sinaps',
    version='0.0.1',
    description='sinaps is a collection of algorithms \
                to simulate single particle tracking experiments',

    author='Clayton Seitz',
    author_email='cwseitz@iu.edu',
    packages=['sinaps'],

    install_requires=['numpy',
                     'trackpy==0.4.2',
                     'pims==0.4.1',
                     'psf',
                     'scikit-image==0.16.2',
                     'seaborn==0.9.0',
                     'matplotlib-scalebar==0.6.1',
                     'matplotlib',
                     'pandas==0.25.3',
                     'scipy',
                     'more-itertools',
                     'scikit-learn'],
)
