from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='MagicONNX',
    version='0.1.0',
    description='Magic modified ONNX model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ronnie_zheng',
    author_email='zl19940307@163.com',
    url='https://gitee.com/Ronnie_zheng/MagicONNX',
    packages=find_packages(),
    package_data={'': ['LICENSE']},
    license='Apache-2.0',
    keywords='deep-learning ONNX',
    install_requires=required,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'License :: Apache-2.0 Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    python_requires='>=3.7'
)
