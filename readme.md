### ycecream -- Never use print() to debug again

Do you ever use `print()` or `log()` to debug your code? Of course you
do. With ycecream, or `y` for short, printing debug information becomes a little smarter.

ycecream is a fork of the popular IceCream module, by Ansgar Grunseid / grunseid.com / grunseid@gmail.com

Main differences with icecream:

* ycecream doesn't depend on ANY external module (i.e. asttokens, six, executing, pyglets, ...)
* ycecream is a single source file package, thus easily installed
* ycecream does not support colouring
* ycecream can switch on/off time inclusion
* ycecream introduces delta time that can also be switched on or off
* ycecream uses PEP8 compatible naming (both in the interface and internal)
* ycecream uses an even shorter name to print (`y` versus `ic` in IceCream)
* ycecrean runs only under Python 3.6 ++

(c)2021 Ruud van der Ham - rt.van.der.ham@gmail.com

### Inspect Variables

Have you ever printed variables or expressions to debug your program? If you've
ever typed something like

```python
print(foo('123'))
```

or the more thorough

```python
print("foo('123')", foo('123'))
```

then `y()` is here to help. With arguments, `y()` inspects itself and prints
both its own arguments and the values of those arguments.

```python
from ycecream import y

def foo(i):
    return i + 333

y(foo(123))
```

Prints

```
y| foo(123): 456
```

Similarly,

```python
d = {'key': {1: 'one'}}
y(d['key'][1])

class klass():
    attr = 'yep'
y(klass.attr)
```

Prints
```
y| d['key'][1]: 'one'
y| klass.attr: 'yep'
```
Just give `y()` a variable or expression and you're done. Easy.


### Inspect Execution

Have you ever used `print()` to determine which parts of your program are
executed, and in which order they're executed? For example, if you've ever added
print statements to debug code like

```python
def foo():
    print(0)
    first()

    if expression:
        print(1)
        second()
    else:
        print(2)
        third()
```

then `y()` helps here, too. Without arguments, `y()` inspects itself and
prints the calling filename, line number, and parent function.

```python
from ycecream import y

def foo():
    y()
    first()
    
    if expression:
        y()
        second()
    else:
        y()
        third()
```

Prints

```
y| example.py:4 in foo()
y| example.py:11 in foo()
```

Just call `y()` and you're done. Simple.


### Return Value

`y()` returns its argument(s), so `y()` can easily be inserted into
pre-existing code.

```pycon
>>> a = 6
>>> def half(i):
>>>     return i / 2
>>> b = half(y(a))
y| a: 6
>>> y(b)
y| b: 3
```


### Miscellaneous

`y.as_str(*args)` is like `y()` but the output is returned as a string instead
of written to stderr.

```pycon
>>> from ycecream import y
>>> s = 'sup'
>>> out = y.as_str(s)
>>> print(out)
y| s: 'sup'
```

Additionally, `y()`'s output can be entirely disabled, and later re-enabled, with
`y.disable()` and `y.enable()` respectively.

```python
from ycecream import y

y(1)

y.disable()
y(2)

y.enable()
y(3)
```

Prints

```
y| 1
y| 3
```

`y()` continues to return its arguments when disabled, of course; no existing
code with `y()` breaks.


### Import Tricks

To make `y()` available in every file without needing to be imported in
every file, you can `install()` it. For example, in a root `A.py`:

```
from ycecream import install
install()

from B import foo
foo()
```

and then in `B.py`, which is imported by `A.py`, just call `y()`:

```
def foo():
    x = 3
    y(x)
```

`install()` adds `y()` to the
[builtins](https://docs.python.org/3.8/library/builtins.html) module,
which is shared amongst all files imported by the interpreter.
Similarly, `y()` can later be `uninstall()`ed, too.

`y()` can also be imported in a manner that fails gracefully if
ycecream isn't installed, like in production environments (i.e. not
development). To that end, this fallback import snippet may prove
useful:

```
try:
    from ycecream import y
except ImportError:  # Graceful fallback if ycecream isn't installed.
    y = lambda *a: None if not a else (a[0] if len(a) == 1 else a)
```
### Configuration

```
y.configure(prefix, output_function, arg_to_string_function,
include_context, include_time, include_delta)`
```
can be used to adopt a custom output prefix (the default is
`y| `), change the output function (default is to write to stderr), customize
how arguments are serialized to strings, and/or include the `y()` call's
context (filename, line number, and parent function) in `y()` output with
arguments.

```
>>> from ycecream import y
>>> y.configure(prefix='hello -> ')
>>> y('world')
hello -> 'world'
```

`prefix` can optionally be a function, too.

```
>>> import time
>>> from ycecream import y
>>>  
>>> def unixTimestamp():
>>>     return '%i |> ' % int(time.time())
>>>
>>> y.configure(prefix=unixTimestamp)
>>> y('world') 
1519185860 |> 'world': 'world'
```

`output_function`, if provided, is called with `y()`'s output instead of that
output being written to stderr (the default).

```pycon
>>> import logging
>>> from ycecream import y
>>>
>>> def warn(s):
>>>     logging.warning(s)
>>>
>>> y.configure(output_function=warn)
>>> y('eep')
WARNING:root:y| 'eep': 'eep'
```

`arg_to_string_function`, if provided, is called with argument values to be
serialized to displayable strings. The default is PrettyPrint's
[pprint.pformat()](https://docs.python.org/3/library/pprint.html#pprint.pformat),
but this can be changed to, for example, handle non-standard datatypes in a
custom fashion.

```pycon
>>> from ycecream import y
>>> 
>>> def toString(obj):
>>>    if isinstance(obj, str):
>>>        return '[!string %r with length %i!]' % (obj, len(obj))
>>>    return repr(obj)
>>> 
>>> y.configure(arg_to_string_function=toString)
>>> y(7, 'hello')
y| 7: 7, 'hello': [!string 'hello' with length 5!]
```

`include_context`, if provided and True, adds the `y()` call's filename, line
number, and parent function to `y()`'s output.

```pycon
>>> from ycecream import y
>>> y.configure(include_context=True)
>>> 
>>> def foo():
>>>   y('str')
>>> foo()
y| example.py:12 in foo()- 'str': 'str'
```

`include_context` is False by default. Note that if you call `y` without any arguments, the context is always shown, regardless of the status `include_context`.

### Compatibility with IceCream

The ycecream module is a fork of IceCream with a number of differences:

* ycecream uses y as the standard interface, whereas IceCream uses ic. To make life easy, ycecream also supports ic!
* icecream colourizes the output by default. This functionality is completely absent in ycebreaker.
* IceCream has many dependencies. On the other hand, ycecream has none
* Icecream requires a number of .py files, whereas ycecream is just one (big) .py file. That makes it possible to use ycecream without even (pip) installing it. Just copy ycecream.py to your work directory.
* In contrast to IceCream, ycecream has a PEP8 (Pythonic) API. Less important for the user, the actual code is also (more) PEP8 compatible.
* With ycecream time inclusion can be controlled independently from context
* A new delta inclusion (time since start of the program) is available in ycecream


### Installation

Installing ycecream with pip is easy.
```
$ pip install ycecream
```
or when you want to upgrade,
```
$ pip install ycecream
```

Alternatively, ycecream.py can be juist copied into you current work directory from GitHub (https://github.com/salabim/ycecream).
