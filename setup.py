from setuptools import setup

setup(
    name='hsfuap',
    version='0.1.0dev',
    author='Dougal J. Sutherland',
    author_email='dougal@gmail.com',
    packages=['hsfuap', 'hsfuap.kernels', 'hsfuap.misc', 'hsfuap.sdm'],
    description='Some miscellaneous utilities I find useful.',
    entry_points={
        'console_scripts': [
            'hsfuap-nystroem = hsfuap.kernels.nystroem:main',
            'hsfuap-sdm-gather = hsfuap.sdm.gather_results:main',
            'hsfuap-sdm-kernelize = hsfuap.sdm.make_kernels:main',
        ],
    },
)
