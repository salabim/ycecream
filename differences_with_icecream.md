
# Differences with IceCream

The ycecream module was originally a fork of IceCream, but has many differences:

```
----------------------------------------------------------------------------------------
characteristic                    ycecream                 IceCream
----------------------------------------------------------------------------------------
colourize                         no                       yes (can be disabled)
platform                          Python 3.6 and higher    Python 2.7, 3.x, PyPy
default name                      y                        ic
dependencies                      none                     many
number of files                   1                        several
usable without installation       yes                      no
can be used as a decorator        yes                      no
can be used as a context manager  yes                      no
PEP (Pythonic) API                yes                      no
sorts dicts                       no, customizable         yes
supports compact, indent and
depth parameters of pprint        yes                      no
use from a REPL                   limited functionality    no
external configuration            via json file            no
observes line_length correctly    yes                      no
indentattion                      4 blanks (overridable)   dependent on length of prefix.
test script                       PyTest                   unittest
----------------------------------------------------------------------------------------
