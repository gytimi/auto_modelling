from setuptools import setup

setup(name='auto_modelling',
      version='0.1',
      description='Auto data preparation and modelling tool.',
      author='Gyorgy Timea',
      author_email='gy.timi44@gmail.com',
	  install_requires=[
          'sklearn',
		  'math',
		  'numpy',
		  'pandas'
      ],
      packages=['auto_modelling'])