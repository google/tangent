from setuptools import setup, find_packages

with open('README.md') as f:
  readme = f.read()

with open('LICENSE') as f:
  lic = f.read()

setup(
    name='tangent',
    version='0.0.1rc1',
    description=('Automatic differentiation using source code transformation '
                 'for Python'),
    long_description=readme,
    author='Google Inc.',
    author_email='noreply@google.com',
    url='https://github.com/google/tangent',
    license=lic,
    packages=find_packages(exclude=('tests')),
    install_requires=[
        'astor>=0.6', 'autograd', 'enum34', 'future', 'gast', 'nose', 'numpy',
        'six', 'tf-nightly>=1.5.0.dev20171026',
    ])
