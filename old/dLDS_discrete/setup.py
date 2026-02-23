import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    author="noga mudrik",

    name="dLDS_discrete_2022",
    version="0.1.01",
    
    author_email="<nmudrik1@jhmi.edu>",
    description="dLDS discrete model package",
    long_description=long_description,
    long_description_content_type="text/markdown",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.8",
    install_requires = ['numpy', 'matplotlib','scipy','scipy','pandas','webcolors',
                        'seaborn','colormap','sklearn', 'pylops','dill','mat73', 'easydev']
)

