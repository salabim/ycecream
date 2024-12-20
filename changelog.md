### changelog | ycecream | sweeter debugging and benchmarking

#### 2024-11-07

This package is not maintained anymore.

It is now called (with slightly different functionality) **peek** and can be found here:

https://pypi.org/project/peek-python/

https://github.com/salabim/peek

https://salabim.org/peek/



#### version 1.3.20  2024-10-21

* serious bug (`__all__` incorrectly defined) in 1.3.19 made that nothing worked properly. Fixed
* test of f-strings now works properly

#### version 1.3.19  2024-10-17

* Python 2.7 is not supported anymore.
* Now uses pprint from Python 3.13, which uses lazy importing of the dataclasses and re packages.

#### version 1.3.18  2024-10-16

* Now compatible with Python >3.12
* Complete  project structure overhaul, including pyproject.toml
* embedder now works on an embedded version as well, so no more need for an unembedded version.
* GitHub update


#### Older versions
```
version 1.3.17  2023-12-04
----------------------------
Removed __all__ from the pprint source, which prevented ycecream to work properly.

version 1.3.16.2  2023-12-02
----------------------------
Changed prepare.py to dynamically make setup.py (from setupx.py and readme.md).

version 1.3.16  2023-11-16
--------------------------
Ycecream now supports the underscore_number functionality of pprint
(even for Python < 3.10, but not Python 2.7).
Therefore, a new attribute underscore_number (abbreviated un) is introduced. 
This attribute is False by default.
Example usage:
    powers = [10**i for i in range(8)]
    y(powers, underscore_numbers=True)
results in
    y| powers: [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]    
    
version 1.3.15  2023-10-06
--------------------------
Updated readme and made it available on PyPI.
Checked compatibility with Python 3.12.


version 1.3.14  2023-01-26
==========================
It is now possible to suppress the parent function/method when line numbers are shown.
To do so, just use one of
    show_line_numbers = "n"
    show_line_numbers = "no parent"
    sln = "n"
    sln = "no parent"
The readme file has been updated and corrected.

version 1.3.13  2023-01-23
==========================
A bug made that there was still a dependency: executing. Fixed.

version 1.3.12  2023-01-23
==========================
A bug in 1.3.11 made that asttokens had to be installed.
Fixed by using a completely different way of embedding the required dependencies,
i.e. with my package embedder (see www.github.com/salabim/package_embedder)

version 1.3.11  2023-01-13
==========================
Added compatibility (0)
-----------------------
Now compatible with Python 3.11
by including the updated executing and _position_node_finder modules
Also updated tests.

version 1.3.10  2022-04-12
==========================
Changed functionality (0)
---------------------
For methods, the class name is also shown in the function name.
Based on a pull request https://github.com/gruns/icecream/pull/120/files


version 1.3.9  2021-06-12
=========================
New functionality (0)
---------------------
A new parameter 'provided' is added to the call y.
If provided is True, output will be generated as usual, but if provided is False,
all output will be suppressed as with enabled=False.
This is useful when output is only wanted if some condition is met, e.g.
    x = 12
    temperature = 18
    y(x, provided=temperature>25)
In this case, nothing will be printed.

New functionality (1)
---------------------
Introduced the method assert_. If the assert_ is called with a Falsy value, an AssertionError will be raised.
The biggest difference with the assert statement is, that this enabling/disabling can be controlled in the program.
E.g.
    x = 12
    temperature = 18
    y.assert_(temperature>25)
This will result in an AssertionError.
but
E.g.
    y.enabled = False
    x = 12
    temperature = 18
    y.assert_(temperature>25)
won't raise an error

Changed functionality (0)
-------------------------
If enabled is False, calling y with as_str will now return the null string.

Changed functionality (1)
-------------------------
The attributes decorator (d) and context_manager (cm) can now only be used in a call to y.
So, they can no longer be part of the configuration .json file, nor are they parameters for new, fork or clone.
This change is made to make sure that decorator (d) and context_manager (dm) can only be applied to the call.


version 1.3.8  2021-05-08
=========================
New functionality (0)
---------------------
Introduced a new attribute "value_only_for_fstrings" (shorthand "voff").
By default, Values_only_for_fstrings is False.
When values_only_for_fstring is True, the f-string itself will not be shown.
Example:
    x = 12.3
    y(f"{x:0.3e}")
    y.values_only_for_fstrings = True
    y(f"{x:0.3e}")
prints
    y| f"{x:0.3e}": '1.230e+01'
    y| '1.230e+01'
Note that, when the test script is run on Python < 3.6, the corresponding test is skipped.

Bugfix (0)
----------
The shortcut st was defined twice, resulting in being a shortcut for show_traceback only.
From now on st is a shortcut for show_time and stb is a shortcut for show_traceback.

Example:
    y.show_time = True
    y(y.st)
    y.show_time = False
    y(y.st)
    y.show_traceback = True
    y(y.stb)
    y.show_traceback = False
    y(y.stb)
prints something like:
    y| @ 14:47:46.356834 ==> y.st: True
    y| y.st: False
    y| y.stb: True
        Traceback (most recent call last)
        File "c:\Users\Ruud\Dropbox (Personal)\Apps\Python Ruud\ycecream\x.py", line 8, in <module>
            y(y.stb)
    y| y.stb: False

More compact output for multiple lines, when the context fits within the wrap_indent on the first line, like
    a = 12
    b = 4
    y.separator = ""
    y(a,b)
    y.prefix = "prefix| "
    y(a,b)
prints
    y|  a: 12
        b: 4
    prefix|
        a: 12
        b: 4



version 1.3.7  2021-05-08
=========================
Changed functionality (0)
-------------------------
More compact output for multiple lines, when the context fits within the wrap_indent on the first line, like
    a = 12
    b = 4
    y.separator = ""
    y(a,b)
    y.prefix = "prefix| "
    y(a,b)
prints
    y|  a: 12
        b: 4
    prefix|
        a: 12
        b: 4

Changed functionality (1)
-------------------------
The yc ycecream object has now "yc| " as prefix and is now different from y. E.g.
    from ycecream y, yc
    a=1
    y(a)
    yc(a)
prints
    y| a: 12
    yc| a: 12
Note that yc is forked from y, so if you change an attribute of y (and that attribute is not
explicitely set in yc)), that attribute is also changed in yc, e.g.
    from ycecream y, yc
    y.enabled = False
    yc(1)
    y.enabled = True
    yc(2)
prints
    yc| 2
and not yc| 1 , as y.enabled propagates to yc.enabled


version 1.3.6  2021-04-29
=========================
Changed functionality (0)
-------------------------
The attribute pair_delimiter is now called separator (shorthand sep).

If separator is blank, multiple items will be automatically placed on multiple lines. E.g.
    from ycecream import y
    a = 12
    b = 4 * ["test"]
    y(a, b)
    y(a, b, sep="")
prints
    y| a: 12, b: ['test', 'test', 'test', 'test']
    y|
        a: 12
        b: ['test', 'test', 'test', 'test']


Changed functionality (1)
-------------------------
The attribute context_delimiter is now called context_separator (shorthand cs).


New functionality (0)
---------------------
Introduced a new attribute: equals_separator (shorthand es).
This string is used to separate the name of a variable of expression and the value (by deafult it is ": ").
Example:
    from ycecream import y
    a = 12
    b = 4 * ["test"]
    y(a, b)
    y(a, b, equals_separator=" => ")
prints
    y| a: 12, b: ['test', 'test', 'test', 'test']
    y| a => 12, b => ['test', 'test', 'test', 'test']


New functionality (1)
---------------------
Ycecream now also has a yc variable that is equal to y. So it is possible to use
    from ycecream import yc
Please note that after
    from ycecream import y
    from ycecream import yc
y and yc refer to the same object!

version 1.3.5  2021-04-20
=========================
New functionality (0)
---------------------
The attribute delta can now be used as an ordinary attribute,
including propagation and initialization from json.

New tests (0)
-------------
Tests for propagation of attributes added.
Tests for delta setting/reading added.

Bugfix (0)
----------
The recently introduced show_traceback facility didn't work under Python 2.7. Fixed.

version 1.3.4  2021-04-16
=========================
New functionality (0)
---------------------
Introduced a new attribute: show_traceback / st .

When show_traceback is True, the ordinary output of y() will be followed by a printout of the
traceback, similar to an error traceback.

    from ycecream import y
    y.show_traceback=True
    def x():
        y()
    
    x()
    x()
prints
    y| #4 in x()
        Traceback (most recent call last)
        File "c:\Users\Ruud\Dropbox (Personal)\Apps\Python Ruud\ycecream\x.py", line 6, in <module>
            x()
        File "c:\Users\Ruud\Dropbox (Personal)\Apps\Python Ruud\ycecream\x.py", line 4, in x
            y()
    y| #4 in x()
        Traceback (most recent call last)
        File "c:\Users\Ruud\Dropbox (Personal)\Apps\Python Ruud\ycecream\x.py", line 7, in <module>
            x()
        File "c:\Users\Ruud\Dropbox (Personal)\Apps\Python Ruud\ycecream\x.py", line 4, in x
            y()

The show_traceback functionality is also available when y is used as a decorator or context manager. 

version 1.3.3  2021-04-14
=========================
New functionality (0)
---------------------
Introduced a new attribute: enforce_line_length / ell .
If True, all output lines will be truncated to the current line_length.
This holds for all output lines, including as_str output.
Example:
    y.configure(enforce_line_length=True, line_length=15)
    s = "abcdefghijklmnopqrstuvwxyz"
    y(s)
    y(show_time=True)
prints something like
    |y|
    |    s: 'abcdefg
    |y| #35 @ 08:14:

New functionality (0)
---------------------
New shorthand alternatives:
    sdi  for sort_dicts
    i    for indent
    de   for depth
    wi   for wrap_indent
    ell  for enforce_line_length

version 1.3.2  2021-04-07
=========================
New functionality (0)
---------------------
y.new() has a new parameter ignore_json that makes it possible to ignore the ycecream.json file.
Thus, if you don't want to use any attributes that are overridden by an ycecream.json file:
    y = y.new(ignore_json=True)

Internal changes (0)
--------------------
The PY2/PY3 specific code is clustered now to make maintenance easier.
In the pprint 3.8 module, the PrettyPrinter class is not even created for PY2,
so no more need to disable specifically several dispatch calls.


version 1.3.1  2021-04-02
=========================
New functionality (0)
---------------------
The ycecream.json file may now also contain shorthand names,
like p for prefix.

New functionality (1)
---------------------
The attribute compact now has a shorthand name: c
So,
    y(d, compact=True)
is equivalent to
    y(d, c=1) 

Changes in install ycecream.py (0)
----------------------------------
install ycecream.py will now also copy ycecream.json to site-packages, if present.


Older versions of Python support (0)
------------------------------------
Ycecream now runs also under Python 2.7.
The implementation required a lot of changes, most notably phasing out pathlib, f-strings,
some syntax incompatibilities and function signatures.
Under Python 2.7, the compact and sort_dicts attributes are ignored as the 2.7 version of
pprint.pformat (which is imported) does not support these parameters. 

Also, the test script needed a lot of changes to be compatible with Python 2.7 and Python 3.x

Under Python 2.7, the scripts
    install ycecream.py
and
    install ycecream from github.py
require pathlib to be installed.

It is most likely that ycecream will run under Python 3.4 and 3.5, but that has not been tested (yet).

version 1.2.1  2021-03-28
=========================
New functionality (0)
---------------------
Assigning delta a value is now alo propogated to any forked instances.
Also the delta value is copied to a cloned version.

Removed functionality (0)
-------------------------
The just introduced way of importing y with
    import ycecream as y
has been removed as it caused a lot of problems and was not very stable.


version 1.2.0  2021-03-26
=========================
New functionality (0)
---------------------
Instead of
    from ycecream import y
it is now possible to use
    import ycecream as y
, which is arguably easier to remember and saves 2 keystrokes: every little bit helps.

The readme now uses the 'import ycecream as y' style, although the
'from ycecream import y' is still available.

New functionality (1)
---------------------
From this version on, it is possible to fork an ycecream instance.
The forked instance will depend on (inherit from) the attributes of the parent instance.
That means that, unless overridden in the forked instance, a change of an attribute in the
parent instance will be propagated into the forked instance. E.g.
    |y1 = y.fork()
    |y2 = y.fork()
    |y2.configure(prefix="y2| ")
    |y1(1000)
    |y2(2000)
    |y.configure(prefix="==>")  # this also affects y1.prefix
    |y1(1001)
    |y2(2001)
prints
    |y| 1000
    |y2| 2000
    |==>1001
    |y2| 2001
Because of this new functionality, the ycecream.enable() functionality is not required anymore
and phased out:
    |y1 = y.fork(show_line_numbers=True, prefix="y1| ")
    |y2 = y.fork(show_delta=True, prefix="y2| ")
    |y1(1000)
    |y2(2000)
    |y.enabled = False
    |y1(1001)
    |y2(2001)
    |y.configure(enabled=True)
    |y1(1002)
    |y2(2002)
prints
    |y1| #16 ==> 1000
    |y2| delta=0.052 ==> 2000
    |y1| #22 ==> 1002
    |y2| delta=0.235 ==> 2002

New functionality (2)
---------------------
A number of shortcuts for attributes have been defined:
    p   prefix
    o   output
    s   serialize
    sln show_line_number
    st  show_time
    sd  show_delta
    se  show_enter
    sx  show_exit
    e   enabled
    ll  line_length
    vo  values_only
    rn  return_none
    d   decorator
    cm  context_manager
So,
    |y.configure(prefix="==>", show_time=True)
is equivalent to
    |y.configure(p="==>", st=True)
This functionality is also present directly on 'attributes':
    |y.prefix = "==>"
is equivalent to
    |y.p = "==>"

Internal changes (0)
====================
Complete overhaul of the attributes and inheritance of instances.
Arguments do not default to None, but to ycecream.nv .
The __init__.py file in PyPI is different from before to support

version 1.1.10  2021-03-20
==========================
New functionality (0)
---------------------
Added as_decorator and as_context_manager parameters to Y.__call__.
These can be used to use 'fast disabling' and still use y() as a decorator or context manager.
So, the table presented with version 1.1.9 should read now:
---------------------------------------------------------------------
                      enabled=True      enabled=False       enabled=[]
---------------------------------------------------------------------
execution speed             normal             normal             fast
y()                         normal          no output        no output
@y                          normal          no output        no output
y(as_decorator=True)        normal          no output        no output
y(as_context_manager=True)  normal          no output        no output
@y()                        normal          no output        TypeError
with y():                   normal          no output   AttributeError
y(as_str=True)              normal             normal           normal
----------------------------------------------------------------------

If you want to use a y as a decorator and still want 'fast disabling':
    |y.configure(enabled=[])
    |@y(as_decorator=True):
    |def add2(x):
    |     return x + 2
    |x34 = add2(30)
prints nothing, whereas
    |y.configure(enabled=[])
    |@y():
    |def add2(x):
    |     return x + 2
    |x34 = add2(30)
would raise a TypeError.
On the other hand
    |y.configure(enabled=False)
    |@y():
    |def add2(x):
    |     return x + 2
    |x34 = add2(30)
would also print nothing. It would however not run as fast as possible for ordinary y() calls.


New functionality (1)
---------------------
Added a new attribute `return_none`, which is False by default.
If True, `y(<arguments>)` will return None, rather than <parameter>. This can be useful when
using ycecream in notebooks. E.g.
    |hello = "world"
    |print(y(hello))
    |y.configure(return_none=True)
    |print(y(hello))
prints
    |y| hello: 'world'
    |world
    |y| hello: 'world'
    |None
    

Improved functionality (0)
--------------------------
Ycecream is now fully compatible with (Jupyter) notebooks.

Improved functionality (1)
--------------------------
When used from a REPL, usage as a decorator is now possible with the as_decorator parameter, like:
    >>>@y(as_decorator)
    >>>def add2(x):
    >>>    return x + 2
    >>>print(add2(x))
    y| called add2(10)
    y| returned 12 from add2(10) in 0.000312 seconds
    12

When used from a REPL, usage as a context manager is now possible with the as_context_manager,like
    >>>with y():
    >>>    pass
    y| enter
    y| exit in 0.000241 seconds

Note that line number are not available in the REPL.

Bug fix (0)
-----------
y.configure() returned the the instance itself rather than None. Fixed.

version 1.1.9  2021-03-17
=========================
New functionality (0)
---------------------
Added the values_only attribute. If False (the default), both the left-hand side (if possible) and the
value will be printed. If True, the left_hand side will be suppressed:
    |hello = "world"
    |y(hello, 2 * hello)
    |y(hello, 2 * hello, values_only=True)
prints
    |y| hello: 'world', 2 * hello = 'worldworld'
    |y| 'world', 'worldworld'
The values_only=True version of y can be seen as a supercharged print/pprint.

New functionality (1)
---------------------
When ycecream is disabled, either via y.configure(enbabled=False) or ycecream.enable(False),
ycecream still has to check for usage as a decorator or context manager, which can be rather time
consuming.
In order to speed up a program with disabled ycecream calls, it is now possible to specify
y.configure(enabled=[]) or ycecream.enabled([]), in which case y will always just return
the given arguments. If ycecream is disabled this way, usage as a @y() decorator  or as a with y():
context manager will raise a runtime error, though. The @y decorator without parentheses will
not raise any exception, though.
Note that calls with as_str=True will not be affected at all by the enabled flag.

The table below shows it all.
---------------------------------------------------------------------
                     enabled=True      enabled=False       enabled=[]
---------------------------------------------------------------------
execution speed            normal             normal             fast
y()                  as specified          no output        no output
@y()                 as specified          no output        TypeError
@y                   as specified          no output        no output
with y():            as specified          no output   AttributeError
y(as_str=True)       as specified       as specified     as specified
---------------------------------------------------------------------


Bug fix (0)
-----------
A file ycecream.json could not be found in the site_packages installation. Fixed.
Added tests to check proper ycecream.json search functionality.

version 1.1.8  2021-03-14
=========================
Added functionality (0)
-----------------------
Ycecream now supports REPLs, be it with limited functionality:
- all arguments are just presented as such, i.e. no left-hand side, e.g.
  >>> hello = "world"
  >>> y(hello, hello * 2)
  y| 'hello', 'hellohello'
  ('hello', 'hellohello')
- line numbers are never shown
- use as a decorator is not supported
- use as a context manager is not supported
  
Bug fix (0)
-----------
Since version 1.1.6, delta was not calculated properly. Fixed.
Added a test to detect this type of bug.


version 1.1.7  2021-03-13
=========================
Added functionality (0)
-----------------------
'wrap_indent' can now be
- a string
- an integer, in which case the indent will be that amount of blanks

Added check for REPL (0)
------------------------
Ycecream now checks explicitely whether it is run from a REPL. If so, a NotImplementedError is raised
immediately at import time.

Bug fix (0)
-----------
Some arguments to y could cause an exception other than ValueError. Fixed.


version 1.1.6  2021-03-10
=========================
New functionality (0)
---------------------
It is now possible to use the compact option of pprint.pformat:
    
    |a = 9 * ['0123456789']
    |y(a)
    |y(a, compact=True)

prints
    |y|
    |    a:
    |        ['0123456789',
    |         '0123456789',
    |         '0123456789',
    |         '0123456789',
    |         '0123456789',
    |         '0123456789',
    |         '0123456789',
    |         '0123456789',
    |         '0123456789']
    |y|
    |    a:
    |        ['0123456789', '0123456789', '0123456789', '0123456789', '0123456789',
    |         '0123456789', '0123456789', '0123456789', '0123456789']
    
New functionality (1)
---------------------
    Also made availabe the indent and depth parameters of pprint.format, so we can now say
    |s="=============================================="
    |a=[40 * "1",[40 * "2",[40 * "3",[40 * "4",[40*'5']]]],40 * "1"]
    |y(a, depth=3,indent=4)
    to get
    |y|
    |    a:
    |        [   '1111111111111111111111111111111111111111',
    |            [   '2222222222222222222222222222222222222222',
    |                ['3333333333333333333333333333333333333333', [...]]],
    |            '1111111111111111111111111111111111111111']

API change (0)
--------------
    The 'show_context' attribute has been changed to 'show_line_number'.

Change of functionality (0)
---------------------------
    The NoSourceCodeAvailable exception has been changed to NotImplementedError with a clear message.

Changed output (0)
------------------
    The line number is now prefixed by a #, whereas the filename is now suppressed if this concerns the current
    program. If it is in another file, the filename will follow the line number, like #12[foo.py]

Bug fix (0)
-----------
    It is now possible to have keyword arguments in functions/methods decorated with y.

Tests (0)
---------
    Added many tests, including testing for line numbers.

Documentation update (0)
------------------------
    The readme file now clearly mentions that the @y is discouraged as it can't reliably detect the source
    when decorating a function definition of more than one line. Instead use @y() that works always as expected.

Internal change (0)
-------------------
    Made the code much more linear, thus avoiding many calls and a clearer structure (IMHO).
    
    Complete overhaul of line numbers and source code. Now all code is cached in a codes dict (per file)
    
    Y.__call__() now always works on a new Y instance that is used in the actual processing, thus making
    the code shorter and more stable
    
    Mass assignment of attributes now via assign function, thus improving stability and shortening code.


version 1.1.5  2021-02-05
=========================
Now, the line_length is properly observed and indentation is not anymore dependent on prefix.
Instead by default 4 blanks (this can overriden with the wrap_indent attribute) are used as an indent.

The attributes context_delimiter and pair_delimiter are now documented.

Exception NoSourceAvailableError is now correctly raised.

Internal change: complete overhaul of formatting.


version 1.1.4  2021-02-03
=========================
Increased the resolution of show_date and show_time (now microseconds).
Delta now start on creation of an Y object and is thus different for each instance.

It is now possible to query the current delta with Y.delta and to reset it with Y.delta = 0

y() can now also be used as a context manager, primarily to time sections of code rather than a function or method:
|   with y():
|       time.sleep(1)
will print
|    y| enter
|    y| exit in 1.000822 seconds

If you also want to see where the context manager was defined:

|   with y(show_context=True):
|       time.sleep(1)
will print something like:
|   y| x6.py:19 ==> enter
|   y| x6.py:19 ==> exit in 1.000544 seconds

As with the decorator y, the show_enter and show_exit parameters may be used to show/hide that information.

The ouput attribute can now also be the string
    "logging.debug", "logging.info", "logging.warning", "logging.error" or "logging.critical"
, which is primarily useful when a configuration is read from ycecream.json .
Note that
    y.configure(logging.info)
ie essentially the same as
    y.configure("logging.info")
    
Internal change: now uses time.perf_counter() to improve accuracy of timings


version 1.1.3  2021-03-02
=========================
There was still a __all__ definition left over, which caused a problem when using ycecream from site-packages
(installed with pip).
Bug fixed.

version 1.1.2  2021-02-28
=========================
It is now possible to use the string 'stderr', 'stdout' and 'null' as output attributes.

Ycecream will now try and read ycecream.json (in the directories in sys.path) to override
the default values, e.g.
    {
    "prefix": "==> ",
    "show_time": true,
    "line_length": 120
    }

Internal change: all defaults are now stored in a class defaults.
Internal change: phased out stderr_print.

version 1.1.1  2021-02-25
=========================
The y decorator now also adds the duration of the function/method call on exit, like
y| returned None from wait(5) in 5.004965 seconds
This can be useful as a basic way to benchmark a function.

version 1.1.0  2021-02-23
=========================
y can now be used as a decorator to debug (keyword) argumnets given to a function.
In connection with this there are two new parameters:
- show_enter
- show_exit

Changed the name of several parameters:
- show_context (was include_context)
- show_time (was include_time)
- show_delta (was include_delta)

The parameter output_function is now called output. And it can now be:
- callable (like before, e.g. print)
- an open text file (e.g. sys.stdout or an explicitely opened file)
- a str or Path object that will be used as the filename (opened with 'a+' for each call)

The parameter arg_to_string is now called serialize.

Complete rewrite of tests, now in PyTest.


version 1.0.2  2021-02-19
=========================
Phased out Y.format() and replaced by Y(as_str=True)

Major update of readme.md

version 1.0.1  2021-02-18
=========================
Several functionality changes

version 1.0.0  2021-02-17
=========================
Initial release version
```
