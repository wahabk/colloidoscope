from setuptools import setup, find_packages

# TODO use setup.py or setup.cfg?

setup(
       # the name must match the folder name 'verysimplemodule'
        name="colloidoscope", 
        version='0.1.0',
        author="Abdelwahab Kawafi",
        author_email="<akawafi3@gmail.com>",
        description='My PhD project to track colloids using confocal and deep learning.',
        # long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'colloidoscope'],

)