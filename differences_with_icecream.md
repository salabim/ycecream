
# Differences with IceCream

The ycecream module was originally a fork of IceCream, but has many differences:

```
----------------------------------------------------------------------------------------
characteristic                    ycecream                 IceCream
----------------------------------------------------------------------------------------
colourize                         no                       yes (can be disabled)
platform                          Python 2.7, >=3.6, PyPy  Python 2.7, >=3.5, PyPy
default name                      y                        ic
dependencies                      none                     many
number of files                   1                        several
usable without installation       yes                      no
can be used as a decorator        yes                      no
can be used as a context manager  yes                      no
PEP8 (Pythonic) API               yes                      no
sorts dicts                       no by default, optional  yes
supports compact, indent and
depth parameters of pprint        yes                      no
use from a REPL                   limited functionality    no
external configuration            via json file            no
observes line_length correctly    yes                      no
indentation                       4 blanks (overridable)   dependent on length of prefix
test script                       pytest                   unittest
----------------------------------------------------------------------------------------
*) sort_dicts and compact are ignored under Python 2.7