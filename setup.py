from setuptools import setup

setup(
    name='entropy',
    version='0.1',
    description='Time series entropy measures implemented with numpy',

    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],
    keywords='entropy fuzzyen sampen',

    url='https://github.com/ixjlyons/entro-py',
    author='Kenneth Lyons',
    author_email='ixjlyons@gmail.com',
    license='new BSD',

    py_modules=['entropy'],

    test_suite='nose.collector',
    tests_require=['nose'],
)
