import re
from .util import doesMatchPattern
from numpy import log as ln
from numpy import log10 as log
from numpy import exp
from numpy import max, min


def math_stream(formula, data, arg, get_data):
    """Internal function to apply math operations as requested on input string
    """
    # Split the user input string at all mathematical operations
    # Allow "( ) * / + -" as math

    pattern = '[\(+\-*^/\)]'
    split_expr = re.split(pattern, formula)

    # Place string literals in dict if cannot be converted to float
    # drop all empty strings from re.split
    quantity_str_dict = dict()

    # Check all stripped strings individually and evaluate
    for i, string in enumerate(split_expr):
        if string != "":
            try:
                float(string)  # Check if string is float
            except:
                # Use math expressions to allow logs and exps
                math_expressions = ['ln', 'log', 'exp', 'max', 'min']

                if string in math_expressions:
                    pass

                else:
                    # Assign generic "val{i}" key to string literal in compliance with
                    # python supported syntax for variables
                    quantity_str_dict[f"val{i}"] = string

    # Parser does not support special string literals (ROIs) due to inproper python variable naming syntax
    # Replace them with generic key as per dictionary
    # Create local variable and assign corresponding data -- needed as eval interprets all input as variable
    # This is indeed local to this function only and cannot be accessed from loadSCAscans
    for k, v in quantity_str_dict.items():
        formula = formula.replace(v, k)

        locals()[k] = get_data(v, data, arg)

    # Return the calculated result
    return eval(formula)