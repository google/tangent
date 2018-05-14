from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
  readme = f.read()

with open('LICENSE') as f:
  lic = f.read()

with open('requirements.txt') as f:
  reqs = list(f.read().strip().split('\n'))

setup(
    name='tangent',
    version='0.1.9',
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
    install_requires=reqs,
)
