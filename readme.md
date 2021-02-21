![logo](./images/logox.png)

### ycecream -- Never use print() to debug again

Do you ever use `print()` or `log()` to debug your code? Of course you
do. With ycecream, or `y` for short, printing debug information becomes a little smarter.

## Installation

Installing ycecream with pip is easy.
```
$ pip install ycecream
```
or when you want to upgrade,
```
$ pip install ycecream --upgrade
```

Alternatively, ycecream.py can be juist copied into you current work directory from GitHub (https://github.com/salabim/ycecream).


## Inspect variables and expressions

Have you ever printed variables or expressions to debug your program? If you've
ever typed something like

```
print(add2(1000))
```

or the more thorough

```
print("add2(1000)", add2(1000)))
```
or (for Python >= 3.8 only):
```
print(f"{add2(1000) =}")
```

then `y()` is here to help. With arguments, `y()` inspects itself and prints
both its own arguments and the values of those arguments.

```
from ycecream import y

def add2(i):
    return i + 2

y(add2(1000))
```

prints
```
y| add2(1000): 1002
```

Similarly,

```
from ycecream import y
class X:
    a = 3
world = {"EN": "world", "NL": "wereld", "FR": "monde", "DE": "Welt"}

y(world, X.a)
```

prints
```
y| world: {'DE': 'Welt', 'EN': 'world', 'FR': 'monde', 'NL': 'wereld'}, X.a: 3
```
Just give `y()` a variable or expression and you're done. Easy.


## Inspect execution

Have you ever used `print()` to determine which parts of your program are
executed, and in which order they're executed? For example, if you've ever added
print statements to debug code like

```
def add2(i):
    print("enter")
    result = i + 2
    print("exit")
    return result
```
then `y()` helps here, too. Without arguments, `y()` inspects itself and
prints the calling filename, line number, and parent function.

```
from ycecream import y
def add2(i):
    y()
    result = i + 2
    y()
    return result
y(add2(1000))
```

prints something like
```
y| x.py:3 in add2()
y| x.py:5 in add2()
```
Just call `y()` and you're done. Simple.


## Return Value

`y()` returns its argument(s), so `y()` can easily be inserted into
pre-existing code.

```
from ycecream import y
def add2(i):
    return i + 2
b = y(add2(1000))
y(b)
```
prints
```
y| add2(1000): 1002
y| b: 1002
```

## Miscellaneous

`y(*args, as_str=True` is like `y(*args)` but the output is returned as a string instead
of written to stderr.

```
from ycecream import y
hello = "world"
s = y.as_str(hello)
print(s)
```
prints
```
y| hello: 'world'
```

Additionally, ycecreams's output can be entirely disabled, and later re-enabled, with
`ycecream.enable(False)` and `ycecream.enable(True)` respectively. The function always returns
the current setting.
Note that this  functions refer to ALL output from ycecream.
```
from ycecream import y, Y, enable

yd = Y(include_delta=True)
y(1)
yd(2)
enable(False)
y(3)
yd(4)
enable(True)
y(5)
yd(6)
print(enable())
```
prints
```
y| 1
y| Δ 0.117 ==> 2
y| 5
y| Δ 0.219 ==> 6
True
```
`y()` continues to return its arguments when disabled, of course.

## Import Tricks

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

`install()` adds `y()` to the builtins module,
which is shared amongst all files imported by the interpreter.
Similarly, `y()` can later be `uninstall()`ed, too.

`y()` can also be imported in a manner that fails gracefully if
ycecream isn't installed, like in production environments (i.e. not
development). To that end, this fallback import snippet may prove
useful:

```
try:
    from ycecream import y
except ImportError:
    y = lambda *args: None if not args else (args[0] if len(a) == 1 else args)
```

## Customization
For the customization, it is important to realize that `y` is an instance of the `ycecream.Y` class, which has
a nuumber of customization attributes:
* `prefix`
* `output_function`
* `arg_to_string_function`
* `include_context`
* `include_time`
* `include_delta`
* `line_wrap_width`
* `pair_delimiter=None`
* `enabled=None`

It is perfectly ok to set/get any of these attributes directly.

But, it is also possible to use a customization directly:
So, it is possible to say
```
from ycecream import y
y(12, prefix="==> ")
```
, which will print
```
==> 12
```
It is also possible to customize y permanently with the customize method. 
```
y.customize(prefix="==> ")
y(12)
```
will print
```
==> 12
```
```
It is possibly easier to say:
```
y.prefix = "==> "
y(12)
```
to print
```
==> 12
```
Yet another way to customize y is by instantiating Y with the required customization:
```
y = Y(prefix="==> ")
y(12)
```
will print
```
==> 12
```
## prefix
```
from ycecream import y
y('world', prefix='hello -> ')
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
y = Y(prefix=unix_timestamp)
y(hello) 
```
prints
```
1613635601 hello: 'world'
```
## output_function
This will allow the output to be handled by something else than the deafult (output being written to stderr).
The output_function should at leae accept one perameter (the one that will be printed)

In the example below, the output is written to stdout.
```
from ycecream import y
y("hello", output_function=print)
```
With
```
from ycecream import y
y = Y(output_function=lambda *args: None)
y("hello")
```
, all output will be suppressed (this can also be done with the enable parameter, see below).

## arg_to_string_function
This will allow to specify how argument values are to be
serialized to displayable strings. The default is pprint, but this can be changed to,
for example, handle non-standard datatypes in a custom fashion.
The arg_to_string function should accept at least one parameter.

```
from ycecream import Y

def add_len(obj):
    if hasattr(obj, "__len__"):
        add = f"[len={len(obj)}]"
    else:
        add = ""
    return f"{repr(obj)} {add}"

y = Y(arg_to_string_function=add_len)
l = list(range(7))
hello = "world"
y(7, hello, l)
```   
prints
```
y| 7 , hello: 'world' [len=5], l: [0, 1, 2, 3, 4, 5, 6] [len=7]
```

## include_context
If True, adds the `y()` call's filename, line number, and parent function to `y()`'s output.

```from ycecream import Y
y = Y(include_context=True)
hello="world"
y(hello)
```
prints something like
```
y| x.py:4 in <module> ==> hello: 'world'
```
Note that if you call `y` without any arguments, the context is always shown, regardless of the status `include_context`.

## include_time
If True, adds the current time to `y()`'s output.

```
from ycecream import Y
y =  Y(include_time=True)
hello="world"
y(hello)
```
prints something like
```
y| @ 13:01:47.588 ==> hello: 'world'
```

## include_delta
If True, adds the number of seconds since the start of the program to `y()`'s output.
```
from ycecream import Y
import time
y = Y(include_delta=True)
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

## line_length
used to specify the line length (for wrapping)

## enable
Can be used to disable the output:

```
from ycecream import y

y.customize(prefix="==> ", enable=False)
world = "perfect"
y(hello)
y.customize(enable=True)
world = "in danger"
```
prints
```
==> hello: world = 'in danger'
```
and nothing about a perfect world.

## sorted_dicts
By default, ycecream does not sort dicts (printed by pprint). However, it is possible to get the
default ppprint behaviour (i.e. sorting dicts) with the sorted_dicts attribute:

```
world = {"EN": "world", "NL": "wereld", "FR": "monde", "DE": "Welt"}
y(world))
s1 = y(world, sort_dicts=False)
s2 = y(world, sort_dicts=True)
```
prints
```
y| world: {'EN': 'world', 'NL': 'wereld', 'FR': 'monde', 'DE': 'Welt'}
y| world: {'EN': 'world', 'NL': 'wereld', 'FR': 'monde', 'DE': 'Welt'}
y| world: {'DE': 'Welt', 'EN': 'world', 'FR': 'monde', 'NL': 'wereld'}
```
### Alternative installation

With `install ycecream.py from github.by`, you can install the ycecream.py directly from GitHub to the site packages (as if it were a pip install).

With `install ycecream.py`, you can install the ycecream.py in your current directory to the site packages (as if it were a pip install).

### Aknowledgement
The ycecream pacakage is a fork of the IceCream package. See https://github.com/gruns/icecream

Many thanks to the author Ansgar Grunseid / grunseid.com / grunseid@gmail.com

### Differences with IceCream

The ycecream module is a fork of IceCream with a number of differences:

* ycecream can't colourize the output (a nice feature of IceCream)
* ycecream runs only on Python 3.6 and higher. (IceCream runs even on Python 2.7).
* ycecream uses y as the standard interface, whereas IceCream uses ic. To make life easy, ycecream also supports ic!
* yceceam has no dependencies. IceCream on the other hand has many (asttoken, colorize, pyglets, ...).
* ycecream is just one .py files, whereas IceCream consits of a number of .py files. That makes it possible to use ycecream without even (pip) installing it. Just copy ycecream.py to your work directory.
* ycecream has a PEP8 (Pythonic) API. Less important for the user, the actual code is also (more) PEP8 compatible. IceCream does not fillow the PEP8 standard.
* ycecream uses a different API to customize (rather than IceCream's configureOutput method
* ycecream time inclusion can be controlled independently from context
* ycecrean has a new delta inclusion (time since start of the program)
* ycecream does not sort dicts by default. This behaviour can be controlled with the sort_dict parameter. (This is implemented by including the pprint 3.8 source code
* ycecream uses pytest for the test scripts rather than YceCream's unittest script.
