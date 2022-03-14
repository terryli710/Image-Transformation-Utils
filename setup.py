from setuptools import setup


setup(
    name="imgtrans",
    version="0.0.1",
    description="image transformation collection",
    url="https://github.com/terryli710/Image-Transformation-Utils",
    author="Yiheng Li",
    license="LICENSE.txt",
    author_email="yiheng@subtlemedical.com",
    python_requires=">=3.7",
    classifiers=[ 'Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'License :: MIT License',],
    install_requires=[
        "numpy",
        "scipy"
    ]
)