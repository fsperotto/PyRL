#you can import setup into the notebook to be able to find SRL package if it is not installed via "pip install srl"
#you shoul use pip install srl
#or pip install -e path/to/srl (editable local installation)

import sys

#necessary for importing local packages
folders = ['../../src/srl/']
for folder in folders:
    if folder not in sys.path:
        sys.path.insert(0, folder)
        