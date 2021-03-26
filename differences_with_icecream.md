
# Differences with IceCream

The ycecream module was originally a fork of IceCream, but has many differences:

```
----------------------------------------------------------------------------------------
characteristic                    ycecream                 IceCream
----------------------------------------------------------------------------------------
colourize                         no                       yes (can be disabled)
platform                          Python >=3.6, PyPy       Python 2.7, >=3.5, PyPy
import statement                  import ycecream as y     from IceCream import ic
alternative import statement      from ycecream import y    
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
hierarchical structure            yes                      no
external configuration            via json file            no
observes line_length correctly    yes                      no
indentation                       4 blanks (overridable)   dependent on length of prefix
test script                       pytest                   unittest
----------------------------------------------------------------------------------------
