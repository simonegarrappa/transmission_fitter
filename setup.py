from setuptools import setup, find_packages

setup(
    name="transmission_fitter",
    version="0.1.0",    
    author="Simone Garrappa",
    author_email="simone.garrappa@gmail.com",
    description="Package for absolute calibration of optical telescopes by fitting the system transmission",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/simonegarrappa/transmission_fitter",  # GitHub repository URL
    packages=find_packages(),  # Automatically find packages
    include_package_data=True,  # Include
    package_data={
        # Specify the data files to include
        "transmission_fitter": ["data/*", "data/Image_Test/*",
                                "data/Image_Test_Coadd/*","data/Templates/*",
                                "data/SourceCatalogs/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Minimum Python version
    install_requires=[
        "numpy", "matplotlib","astropy","pandas","lmfit","gaiaxpy",
          "catsHTM","scipy","seaborn","astroquery" 
    ]
)
