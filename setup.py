from setuptools import setup, find_packages

setup(
       # the name must match the folder name 'verysimplemodule'
        name="colloidoscope", 
        version='0.1.0',
        author="Abdelwahab Kawafi",
        author_email="<akawafi3@gmail.com>",
        description='My PhD project to track colloids using confocal and deep learning.',
        # long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
                'numpy>=1.19',
                'torch',
                'matplotlib',
                'pathlib2',
                'trackpy',
                'numba',
                'pandas',
                'matplotlib',
                'seaborn',
                'lxml',
                'h5py',
                'napari',
                'PyQt5',
                'scipy',
                'scikit-learn',
                'scikit-image',
                'torchio',
                'neptune-client',
                'ray[tune]',
                'tqdm',
                'psf',
                'perlin-numpy @ git+https://github.com/pvigier/perlin-numpy',
                # 'monai @ git+https://github.com/wahabk/MONAI'
        ],
        keywords=['python', 'colloidoscope'],
)