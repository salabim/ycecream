### ycecream -- Never use print() to debug again
![plot](./images/logo_small.png)
```
 _   _   ___   ___   ___  _ __   ___   __ _  _ __ ___
| | | | / __| / _ \ / __|| '__| / _ \ / _` || '_ ` _ \
| |_| || (__ |  __/| (__ | |   |  __/| (_| || | | | | |
 \__, | \___| \___| \___||_|    \___| \__,_||_| |_| |_|
 |___/     The Pythonic, no depencency fork of IceCream

Do you ever use `print()` or `log()` to debug your code? Of course you
do. With ycecream, or `y` for short, printing debug information becomes a little smarter.

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
    def configure(
        self,
        prefix=None,
        output_function=None,
        arg_to_string_function=None,
        include_context=None,
        include_time=None,
        include_delta=None,
        line_wrap_width=None,
        pair_delimiter=None,
        enabled=None,
    ):
```
can be used to adopt a custom output prefix (the default is
`y| `), change the output function (default is to write to stderr), customize
how arguments are serialized to strings, and/or include the `y()` call's
context (filename, line number, and parent function) in `y()` output with
arguments.

```
from ycecream import y
y.configure(prefix='hello -> ')
y('world')
```
prints
```
hello -> 'world'
```

`prefix` can optionally be a function, too.

```
import time
from ycecream import y
def unix_timestamp():
    return f"{int(time.time())} "
hello = "world"
y.configure(prefix=unix_timestamp)
y(hello) 
```
prints
```
1613635601 hello: 'world'
```

`output_function`, if provided, is called with `y()`'s output instead of that
output being written to stderr (the default).
In the example below, the output is written to stdout.
```
from ycecream import y
y.configure(output_function=print)
y('hello')
```
With
```
from ycecream import y

y.configure(output_function=lambda *args: None)
y('hello')
```
, all output will be suppressed (this van also be done with disable, see below).

`arg_to_string_function`, if provided, is called with argument values to be
serialized to displayable strings. The default is PrettyPrint's
[pprint.pformat()](https://docs.python.org/3/library/pprint.html#pprint.pformat),
but this can be changed to, for example, handle non-standard datatypes in a
custom fashion.

```
from ycecream import y

def add_len(obj):
    if hasattr(obj, "__len__"):
        add = f"[len={len(obj)}]"
    else:
        add = ""
    return f"{repr(obj)} {add}"

y.configure(arg_to_string_function=add_len)
l = list(range(7))
hello = "world"
y(7, hello, l)
```   
prints
```
y| 7 , hello: 'world' [len=5], l: [0, 1, 2, 3, 4, 5, 6] [len=7]
```

`include_context`, if provided and True, adds the `y()` call's filename, line
number, and parent function to `y()`'s output.

```from ycecream import y
y.configure(include_context=True)
hello="world"
y(hello)
```
prints something like
```
y| x.py:4 in <module> ==> hello: 'world'
```
`include_context` is False by default. Note that if you call `y` without any arguments, the context is always shown, regardless of the status `include_context`.

`include_time`, if provided and True, adds the current time to `y()`'s output.

```
from ycecream import y
y.configure(include_time=True)
hello="world"
y(hello)
```
prints something like
```
y| @ 13:01:47.588 ==> hello: 'world'
```
`include_delta`, if provided and True, adds the number of seconds since the start of the program to `y()`'s output.
```
from ycecream import y
import time
y.configure(include_delta=True)
hello="world"
y(hello)
time.sleep(1)
y(hello)
```
prints something like
```
y| Δ 0.021 ==> hello: 'world'
y| Δ 1.053 ==> hello: 'world'
```
Of course, it is possible to use several includes at the same time:
```
from ycecream import y
y.configure(include_context=True, include_time=True, include_delta=True)
hello="world"
y(hello)
```
, which will print something like
```
y| x.py:4 in <module> @ 13:08:46.200 Δ 0.030 ==> hello: 'world'
```




### Aknowledgement
The ycecream pacakage is a fork of the IceCream package. See https://github.com/gruns/icecream

Many thanks to the author Ansgar Grunseid / grunseid.com / grunseid@gmail.com

### Copyright
(c)2021 Ruud van der Ham - rt.van.der.ham@gmail.com

### Differences with IceCream

The ycecream module is a fork of IceCream with a number of differences:

* ycecream can't colourize the output (a nice feature of IceCream)
* ycecream runs only on Python 3.6 and higher. (IceCream runs even on Python 2.7).
* ycecream uses y as the standard interface, whereas IceCream uses ic. To make life easy, ycecream also supports ic!
* yceceam has no dependencies. IceCream on the other hand has many (asttoken, colorize, pyglets, ...).
* ycecream is just one .py files, whereas IceCream consits of a number of .py files. That makes it possible to use ycecream without even (pip) installing it. Just copy ycecream.py to your work directory.
* ycecream has a PEP8 (Pythonic) API. Less important for the user, the actual code is also (more) PEP8 compatible. IceCream does not fillow the PEP8 standard.
* ycecream time inclusion can be controlled independently from context
* ycecrean has a new delta inclusion (time since start of the program)

### Installation

Installing ycecream with pip is easy.
```
$ pip install ycecream
```
or when you want to upgrade,
```
$ pip install ycecream --upgrade
```

Alternatively, ycecream.py can be juist copied into you current work directory from GitHub (https://github.com/salabim/ycecream).
