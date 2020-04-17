from setuptools import setup, find_packages

import sys
print(sys.version_info)

if sys.version_info < (3,5):
    sys.exit('Python version 3.5 or later required. Exiting.')

setup(name='nmpc_system_optimizer_lyapunov',
   version='0.1',
   python_requires='>=3.5',
   description='A templating framework for acados',
   url='http://github.com/zanellia/nmpc_system_optimizer_lyapunov',
   author='Andrea Zanelli',
   license='BSD',
   packages = find_packages(),
   include_package_data = True,
   setup_requires=['setuptools_scm'],
   use_scm_version={
     "fallback_version": "0.1-local",
     "root": "../..",
     "relative_to": __file__
   },
   install_requires=[
      'numpy',
      'scipy',
      'casadi==3.5.1',
      'matplotlib',
      'control',
      'slycot'
   ],
   zip_safe=False
)
