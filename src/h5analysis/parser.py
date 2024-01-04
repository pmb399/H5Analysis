# Import re search module
import re

def parse(formula):
    """Internal function to apply math operations as requested on input string

        Parameters
        ----------
        formula: string

        Returns
        -------
        constituents: list
            all specified data streams
    """

    # Split the user input string at all mathematical operations
    # Allow "( ) * / + -" as math

    # Split string for specified pattern
    constituents = list()
    pattern = '[\(+\-*^/\)]'
    split_expr = re.split(pattern, formula)

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

                else: # String is a data stream
                    constituents.append(string)

    return constituents