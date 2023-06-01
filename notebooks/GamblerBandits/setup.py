import sys

#necessary for importing local packages
folders = ['../../src/srl/mab/']
for folder in folders:
    if folder not in sys.path:
        sys.path.insert(0, folder)