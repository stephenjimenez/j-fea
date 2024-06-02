# j-fea

Jiménez Finite Element Analysis

The purpose of this tool is to demonstrate how to set up and solve the 2D linear elasticity problem using the finite element method. This solver is intended for education purposes and is not optimized.

This code relies on the `numpy` package. It also uses type hinting and can thus only run on Python 3.5+.

## Execution

To run this code, enter the `j-fea` directory and execute:

```bash
python -B __main__.py
```

You can also run this code from outside the `j-fea` directory by executing:

```bash
python -B j-fea
```

Python will automatically select `__main__.py` as the starting point for code executing.

## Testing

To run the unit tests, execute:
```bash
python -B -m unittest
```
Ignore `-B` parameter if you don't mind generating `__pycache__` directories everywhere.
