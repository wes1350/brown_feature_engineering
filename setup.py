import setuptools

setuptools.setup(
    name='brown_feature_engineering',
    version='0.0.1',
    author='Brown',
    maintainer='Wesley Runnels',
    maintainer_email='wrunnels@mit.edu',
    description='Package for feature engineering',
    license='MIT',
    packages=setuptools.find_packages(),
    url='https://gitlab.datadrivendiscovery.org/wrunnels/brown_feature_engineering.git',
    entry_points={
        'd3m.primitives': [
            'data_transformation.feature_transform.Brown = feature_engineering.data_transformation:DataframeTransform'
        ]
    },
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSE Approved :: MIT License',
                 'Operating System :: OS Independent']
)
