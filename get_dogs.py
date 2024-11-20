# list_functions.py

import datasets 
from datasets import *
import inspect

def has_dog(column):
    for row in column:
        if "dog" in row:
            return row
    return False


# Iterate through all the names in the module
for name, obj in inspect.getmembers(datasets):
    # Check if the object is a function and starts with capital letter
    if inspect.isfunction(obj) and name[0].isupper():
        dataset =  globals()[name]() # run function with the same name
        captions = dataset['caption'].dropna()
        dog = has_dog(captions)
        if dog:
            print(f"Dataset {name} has a dog in the caption: {dog}")
        else:
            print(f"Dataset {name} does not have a dog in the caption")
