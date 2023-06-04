from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

#with open("requirements.txt", "r") as fh:
#    requirements = fh.read()
#    requirements = requirements.split()

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'SMPyBandits',
    'tqdm'
]

setup(
    name="pyrl",
    version="0.0.1",
    author="Filipo Studzinski Perotto, Aymane Ouhabi, Melvine Nargeot",
    author_email="filipo.perotto@onera.fr",
    description="Safe and Survival Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fsperotto/pyrl",
    #license = 'MIT',
    package_dir = {"": "src"},
    #packages=find_packages(),
	packages=['pyrl', 'pyrl.mab', 'pyrl.mdp'],
    #packages=find_packages(exclude=['data', 'notebooks']), 
    #packages=find_namespace_packages(include=['smab'],exclude=['extra', 'old', 'notebooks']),
    #include_package_data = True,
    #package_data={'corpus': ['corpus']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    #install_requires = requirements,
    #tests_require = [],    
)