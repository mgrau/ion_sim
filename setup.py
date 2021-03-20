from setuptools import setup, find_packages

setup(name='ion_sim',
      version='0.1.0',
      description='solve for ion motion in arbitrary potentials',
      url='https://github.com/mgrau/ion_sim',
      author='Matt Grau',
      author_email='matt.grau@gmail.com',
      python_requires='>=3.6',
      install_requires=[
              'numpy',
              'scipy',
              'pint',
              'autograd'
      ],
      packages=find_packages())
