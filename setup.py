from setuptools import setup, find_packages

setup(
    name='SAYLLH',
    version='1.0.0',
    author='A. Schneider, C. Arguelles, and T. Yuan',
    author_email='aschneider@icecube.wisc.edu, caad@mit.edu, and tyuan@icecube.wisc.edu',
    description='Package implements the SAY likelihood formalism for incorporating monte carlo uncertainties in bin likelihood problems as discussed in arXiv:XXXX.XXXX.',
    long_description=open('README.md').read(),
    url='https://github.com/hogenshpogen/SAYLikelihood.git',
    packages=find_packages('./'),
    package_data={
        'SAYLLH':['resources/logo/SAYLLHLogo.png'],
    },
    install_requires=['numpy',
                      'scipy',],
    extras_require={
        'plotting':  ['matplotlib'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
    )
