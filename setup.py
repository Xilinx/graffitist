from setuptools import setup, find_packages

from os import path

pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='graffitist',
      version='2.0.2',
      description='Graph Transforms to Quantize and Retrain Deep Neural Nets in TensorFlow.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Sambhav R. Jain',
      author_email='sambhavj@xilinx.com',
      url='https://github.com/Xilinx/graffitist',
      license='BSD 3-Clause',
      python_requires='~=3.6',
      packages=find_packages(exclude=['docs', 'tests', 'scripts', 'datasets', 'models*']),
      package_data={'graffitist': ['kernels/*.so']},
      zip_safe=False)
