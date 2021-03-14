
# Differences with IceCream

The ycecream module was originally a fork of IceCream, but has many differences:

* ycecream can't colourize the output
* ycecream runs only on Python 3.6 and higher
* ycecream uses y as the standard interface, whereas IceCream uses ic.
* yceceam has no dependencies. IceCream on the other hand has many (asttoken, colorize, pyglets, ...).
* ycecream is just one .py file, whereas IceCream consists of a number of .py files. That makes it possible to use ycecream without even (pip) installing it. Just copy ycecream.py to your work directory.
* ycecream can be used as a decorator of a function showing the enter and/or exit event as well as the duration.
* ycecream can be used as a context manager to benchmark code.
* ycecream has a PEP8 (Pythonic) API. Less important for the user, the actual code is also fully PEP8 compatible.
* ycecream uses a different API to configuration (rather than IceCream's configureOutput method)
* ycecream can toggle line number, time and delta inclusion independently
* ycecream does not sort dicts by default. This behaviour can be controlled with the `sort_dict` parameter. (This is implemented by including the pprint 3.8 source code)
* ycecream can use the compact, indent and depath parameters of pprint to allow for more formatting flexibility
* ycecream can be used in REPL (with very lmited functionality)
* ycecream can be configured from a json file, thus overriding some or all default settings at import time.
* ycecream has a line_length attribute that is observed correctly.
* ycecream indents output by 4 blanks (overridable) rather than IceCream's indentation that depends on the length of prefix.
* ycecream uses pytest for the test scripts rather than IceCream's unittest script.
