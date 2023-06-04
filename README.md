# PyRL

PyRL is a Python Platform for Reinforcement Learning that integrates with many other state of the art RL packages, including the Deep Neural Network based methods.

It is specially designed to include Safe and Survival Reinforcement Learning methods, but also Multi-Armed Bandits, Stochastic Gambling Processes, Dynamic Programming, Classical Planning.


## Gambling Processes

Python scripts and Jupyter notebooks concerning: Pascal's Triangle, Catalan's Triangle, Gambler's Ruin, Decisional Gambling Processes, Multi-Armed Bandits, Survival Bandits, Multi-Armed Gambler, Survival Decision Processes, and Reinforcement Learning


## Project Structure

pyrl                        --> project_name

├── docs                    --> auto docs using sphinx

│   ├── make.bat            --> build docs from source into build (windows)

│   ├── Makefile            --> build docs from source into build (linux)

│   ├── build

│       └── index.html

│   └── source

│       ├── conf.py

│       └── index.rst

├── examples                --> .py use case examples

│   └── *.py

├── notebooks               --> use cases as notebooks

│   └── *.ipynb

├── src                     --> SCRIPTS SOURCE CODE

│   └── pyrl                --> package_name

        ├── gr.py
		
        ├── gr.py

│       └── __init__.py

├── tests

#│   └── test_srl             --> test folder for package

#│       └── __init__.py

├── .gitignore

├── .readthedocs.yml

#├── .travis.yml  

├── AUTHORS.txt

├── LICENSE.txt

├── MANIFEST.in					--> list of data files to be included

├── README.md

├── requirements.txt

├── pyproject.toml

├── setup.cfg

├── setup.py

#└── tox.ini