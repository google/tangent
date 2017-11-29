from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
  readme = f.read()

with open('LICENSE') as f:
  lic = f.read()

setup(
    name='tangent',
    version='0.1.8',
    description=('Automatic differentiation using source code transformation '
                 'for Python'),
    long_description=readme,
    author='Google Inc.',
    author_email='alexbw@google.com',
    url='https://github.com/google/tangent',
    license=lic,
    packages=find_packages(exclude=('tests')),
    package_data={'':['README.md','LICENSE']},
    keywords=[
        'autodiff', 'automatic-differentiation', 'machine-learning',
        'deep-learning'
    ],
    install_requires=[
        'astor>=0.6',
        'autograd>=1.2',
        'enum34',
        'future',
        'gast',
        'nose',
        'numpy',
        'six',
        'tf-nightly>=1.5.0.dev20171026',
    ])
