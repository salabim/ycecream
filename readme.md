![logo](./images/logox.png)


Do you ever use `print()` or `log()` to debug your code? If so,  ycecream, or `y` for short, will make printing debug information a lot sweeter.

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
y| world: {"EN": "world", "NL": "wereld", "FR": "monde", "DE": "Welt"}, X.a: 3
```
Just give `y()` a variable or expression and you're done. Sweet, isn't it?


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
y| add2(1000): 1002
```
Just call `y()` and you're done. Isn't that sweet?


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
## Debug entry and exit of function calls

When you apply `y` as a decorator to a function or method, both the entry and exit can be tracked.
The (keyword) arguments passed will be shown and upon return, the return value.

```
from ycecream import y
@y
def mul(x, y):
    return x * y
    
print(mul(5, 7))
```
prints
```
y| called mul(5, 7)
y| returned 35 from mul(5, 7) in 0.000006 seconds
35
```
It is possible to suppress the print-out of either the enter or the exit information with
the show_enter and show_exit parameters, like:

```
from ycecream import y
@y(show_exit=False)
def mul(x, y):
    return x * y
    
print(mul(5, 7))
```
prints
```
y| called mul(5, 7)
35
```

The included duration upon exit offers basic function benchmark functionality.

For instance, with
```
from ycecream import y
import time

@y(show_enter=False)
def do_sort(n):
    x = sorted(list(range(10 ** n)))
        
for i in range(8):
    do_sort(i)
```
the ouput will show the effects of the population size on the sort speed:
```
y| returned None from do_sort(0) in 0.000011 seconds
y| returned None from do_sort(1) in 0.000032 seconds
y| returned None from do_sort(2) in 0.000010 seconds
y| returned None from do_sort(3) in 0.000042 seconds
y| returned None from do_sort(4) in 0.000716 seconds
y| returned None from do_sort(5) in 0.004501 seconds
y| returned None from do_sort(6) in 0.049840 seconds
y| returned None from do_sort(7) in 0.490177 seconds
```

## Miscellaneous

`y(*args, as_str=True)` is like `y(*args)` but the output is returned as a string instead
of written to stderr.

```
from ycecream import y
hello = "world"
s = y.as_str(hello)
print(s, end="")
```
prints
```
y| hello: 'world'
```

Additionally, ycecreams's output can be entirely disabled, and optionally  later re-enabled, with
`ycecream.enable(False)` and `ycecream.enable(True)` respectively. The function always returns
the current (new) setting.
Note that this function refers to ALL output from ycecream.
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
a number of customization attributes:
* `prefix`
* `output`
* `serialize`
* `show_context`
* `show_time`
* `show_delta`
* `show_enter`
* `show_exit`
* `line_wrap_width`
* `pair_delimiter=None`
* `enabled=None`

It is perfectly ok to set/get any of these attributes directly.

But, it is also possible to use a customization in the call to `y`:
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
It is arguably easier to say:
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

`prefix` can also be a function, too.

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

## output
This will allow the output to be handled by something else than the default (output being written to stderr).

The `output` attribute can be

* a callable that accepts at least one parameter (the text to be printed)
* a string or Path object that will be used as the filename
* the null string (""), resulting in a dummy action when a text has to be printed
* a text file that is open for writing/appending

In the example below, 
```
from ycecream import y
import sys
y(1, output=print)
y(2, output=sys.stdout
with open("test", "a+") as f:
    y(3, output=f)
y(4, output="")
```
* `y| 1` will be printed to stdout
* `y| 2` will be printed to stdout
* `y| 3` will be appended to the file test
* `y| 4` will *disappear*

## serialize
This will allow to specify how argument values are to be
serialized to displayable strings. The default is pprint, but this can be changed to,
for example, to handle non-standard datatypes in a custom fashion.
The serialize function should accept at least one parameter.

```
from ycecream import Y

def add_len(obj):
    if hasattr(obj, "__len__"):
        add = f" [len={len(obj)}]"
    else:
        add = ""
    return f"{repr(obj)}{add}"

y = Y(serialize=add_len)
l = list(range(7))
hello = "world"
y(7, hello, l)
```   
prints
```
y| 7, hello: 'world' [len=5], l: [0, 1, 2, 3, 4, 5, 6] [len=7]
```

## show_context
If True, adds the `y()` call's filename, line number, and parent function to `y()`'s output.

```from ycecream import Y
y = Y(show_context=True)
hello="world"
y(hello)
```
prints something like
```
y| x.py:4 in <module> ==> hello: 'world'
```
Note that if you call `y` without any arguments, the context is always shown, regardless of the status `include_context`.

## show_time
If True, adds the current time to `y()`'s output.

```
from ycecream import Y
y =  Y(show_time=True)
hello="world"
y(hello)
```
prints something like
```
y| @ 13:01:47.588 ==> hello: 'world'
```

## show_delta
If True, adds the number of seconds since the start of the program to `y()`'s output.
```
from ycecream import Y
import time
y = Y(show_delta=True)
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
world = "on fire"
```
prints
```
==> hello: world = 'on fire'
```
and nothing about a perfect world.

## sort_dicts
By default, ycecream does not sort dicts (printed by pprint). However, it is possible to get the
default pprint behaviour (i.e. sorting dicts) with the sorted_dicts attribute:

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
* ycecream is just one .py file, whereas IceCream consists of a number of .py files. That makes it possible to use ycecream without even (pip) installing it. Just copy ycecream.py to your work directory.
* ycecream can be used as a decorator of a function showing the enter and/or exit event as well as the duration
* ycecream has a PEP8 (Pythonic) API. Less important for the user, the actual code is also (more) PEP8 compatible. IceCream does not follow the PEP8 standard.
* ycecream uses a different API to customize (rather than IceCream's configureOutput method)
* ycecream time showing can be controlled independently from context
* ycecream can optionally show a delta (time since start of the program)
* ycecream uses a pytest script rather than IceCream's unittest script
* ycecream does not sort dicts by default. This behaviour can be controlled with the sort_dict parameter. (This is implemented by including the pprint 3.8 source code)
* ycecream uses pytest for the test scripts rather than IceCream's unittest script.
