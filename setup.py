from setuptools import setup

def readme():
    if open('README.md'):
        with open('README.md') as f:
            return f.read()
    else:
        return None
    
setup(
    name='datasol',
    version = '0.0.1',
    description ='Anomaly Detetion',
    #long_description=readme(),
    #long_description_content_type= 'text/markdown',
    classifiers=['Development Status :: 5 - Production/Stable',
    'License :: MIT License',
    'Programming Language :: Python :: 3.9'
    ],
    url='',
    author='Maxim Melichov',
    author_email='maxme006@gmail.com',
    license= 'MIT',
    # packages=['roconfiguration'],
    # install_requires=['Pandas,Numpy,matplotlib,sklearn,tensorflow,warnings'],
    # include_package_data=True,
    zip_safe=False,
    py_modules=["datasol"],
    package_dir={'':'src'},
)