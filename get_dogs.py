# list_functions.py

import datasets 
from datasets import *
import inspect

def has_dog(column):
    for row in column:
        if "dog" in row:
            return row
    return False

def has_cat_flap(column):
    for row in column:
        if "cat flap" in row:
            return row

def has_ice_cube(column):
    for row in column:
        if "ice cube tray" in row:
            return row

def has_coffee_maker(column):
    for row in column:
        if "coffee maker lid" in row:
            return row


# Iterate through all the names in the module
for name, obj in inspect.getmembers(datasets):
    # Check if the object is a function and starts with capital letter
    if inspect.isfunction(obj) and name[0].isupper():
        dataset =  globals()[name]() # run function with the same name
        captions = dataset['caption'].dropna()
        cat = has_cat_flap(captions)
        if cat:
            print(f"Dataset {name} has a cat in the caption: {cat}")
        else:
            print(f"Dataset {name} does not have a cat in the caption")
        ice = has_ice_cube(captions)
        if ice:
            print(f"Dataset {name} has a ice in the caption: {ice}")
        else:
            print(f"Dataset {name} does not have a ice in the caption")
        coffee = has_coffee_maker(captions)
        if coffee:
            print(f"Dataset {name} has a coffee in the caption: {coffee}")
        else:
            print(f"Dataset {name} does not have a coffee in the caption")
