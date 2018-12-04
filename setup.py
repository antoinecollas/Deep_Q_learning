from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()
setup(
    name='dql',
    version='0.1',
    description='Deep Q learning implementation',
    license='MIT',
    long_description=long_description,
    author='Antoine Collas',
    #author_email='foomail@foo.com',
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
#    install_requires=['bar', 'greek'], #external packages as dependencies
)