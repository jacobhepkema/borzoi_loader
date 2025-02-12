import sys
from setuptools import setup
from setuptools import find_packages


def setup_package():
    install_requires = ['numpy<2', 'pandas', 'pyBigWig', 'enformer-pytorch']
    metadata = dict(
        name='enformer_loader',
        version='0.0.1',
        description='enformer_loader',
        url='https://github.com/jacobhepkema/enformer_loader',
        author='Jacob Hepkema',
        author_email='jacob.hepkema@sanger.ac.uk',
        license='MIT License',
        packages=find_packages(),
        install_requires=install_requires)

    setup(**metadata)


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        sys.exit('Sorry, Python < 3.0 is not supported')

    setup_package()
