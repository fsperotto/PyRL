import sys

#necessary for importing local packages
folders = ['../../packages/smab/smab/', '../../packages/mymab/']
for folder in folders:
    if folder not in sys.path:
        sys.path.insert(0, folder)