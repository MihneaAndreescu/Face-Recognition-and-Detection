import os
from fnmatch import fnmatch

root = "."
pattern = "*.py"

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            print(os.path.join(path, name))