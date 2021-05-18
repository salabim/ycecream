from __future__ import print_function

#   _   _   ___   ___   ___  _ __   ___   __ _  _ __ ___
#  | | | | / __| / _ \ / __|| '__| / _ \ / _` || '_ ` _ \
#   \__, || (__ |  __/| (__ | |   |  __/| (_| || | | | | |
#   |___/  \___| \___| \___||_|    \___| \__,_||_| |_| |_|
#                       sweeter debugging and benchmarking

__version__ = "1.3.8"

"""
See https://github.com/salabim/ycecream for details

(c)2021 Ruud van der Ham - rt.van.der.ham@gmail.com

Inspired by IceCream "Never use print() to debug again".
Also contains some of the original code.
IceCream was written by Ansgar Grunseid / grunseid.com / grunseid@gmail.com
"""


import inspect
import sys
import datetime
import time
import textwrap
import contextlib
import functools
import json
import logging
import collections
import numbers
import ast
import os
import copy
import traceback

nv = object()

PY2 = sys.version_info.major == 2
PY3 = sys.version_info.major == 3

if PY2:

    def perf_counter():
        return time.time() if _fixed_perf_counter is None else _fixed_perf_counter

    class Path(object):
        pass

    import io

    def ycecream_pformat(obj, width, indent, depth):
        return pformat(obj, width=width, indent=indent, depth=depth).replace("\\n", "\n")

    text_type = unicode
    binary_type = str

    def iteritems(d, **kwargs):
        return d.iteritems(**kwargs)


if PY3:

    def perf_counter():
        return time.perf_counter() if _fixed_perf_counter is None else _fixed_perf_counter

    from pathlib import Path

    def ycecream_pformat(obj, width, compact, indent, depth, sort_dicts):
        return pformat(obj, width=width, compact=compact, indent=indent, depth=depth, sort_dicts=sort_dicts).replace("\\n", "\n")

    text_type = str
    binary_type = bytes
    xrange = range

    def iteritems(d, **kwargs):
        return iter(d.items(**kwargs))


class Default(object):
    pass


default = Default()


def change_path(new_path):  # used in tests
    global Path
    Path = new_path


_fixed_perf_counter = None


def fix_perf_counter(val):  # for tests
    global _fixed_perf_counter
    _fixed_perf_counter = val


shortcut_to_name = {
    "p": "prefix",
    "sln": "show_line_number",
    "st": "show_time",
    "sd": "show_delta",
    "sdi": "sort_dicts",
    "se": "show_enter",
    "sx": "show_exit",
    "stb": "show_traceback",
    "e": "enabled",
    "ll": "line_length",
    "c": "compact",
    "i": "indent",
    "de": "depth",
    "wi": "wrap_indent",
    "cs": "context_separator",
    "sep": "separator",
    "es": "equals_separator",
    "vo": "values_only",
    "voff": "values_only_for_fstrings",
    "rn": "return_none",
    "ell": "enforce_line_length",
    "d": "decorator",
    "cm": "context_manager",
    "dl": "delta",
}


def set_defaults():
    default.prefix = "y| "
    default.output = "stderr"
    default.serialize = ycecream_pformat  # can't use pformat directly as that is defined later
    default.show_line_number = False
    default.show_time = False
    default.show_delta = False
    default.sort_dicts = False
    default.show_enter = True
    default.show_exit = True
    default.show_traceback = False
    default.enabled = True
    default.line_length = 80
    default.compact = False
    default.indent = 1
    default.depth = 1000000
    default.wrap_indent = "    "
    default.context_separator = " ==> "
    default.separator = ", "
    default.equals_separator = ": "
    default.values_only = False
    default.values_only_for_fstrings = False
    default.return_none = False
    default.enforce_line_length = False
    default.one_line_per_pairenforce_line_length = False
    default.decorator = False
    default.context_manager = False
    default.start_time = perf_counter()


def apply_json():
    ycecream_name = "ycecream"

    config = {}
    for path in sys.path:
        json_file = os.path.join(path, ycecream_name + ".json")
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                config = json.load(f)
            break
        json_dir = os.path.join(path, ycecream_name)
        json_file = os.path.join(json_dir, ycecream_name + ".json")
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                config = json.load(f)
            break

    for k, v in config.items():

        if k in ("serialize", "start_time"):
            raise ValueError("error in {json_file}: key {k} not allowed".format(json_file=json_file, k=k))

        if k in shortcut_to_name:
            k = shortcut_to_name[k]
        if hasattr(default, k):
            setattr(default, k, v)
        else:
            if k == "delta":
                setattr(default, "start_time", perf_counter() - v)
            else:
                raise ValueError("error in {json_file}: key {k} not recognized".format(json_file=json_file, k=k))


def no_source_error(s=None):
    if s is not None:
        print(s)  # for debugging only
    raise NotImplementedError(
        """
Failed to access the underlying source code for analysis. Possible causes:
- decorated function/method definition spawns more than one line
- used from a frozen application (e.g. packaged with PyInstaller)
- underlying source code was changed during execution"""
    )


def return_args(args, return_none):
    if return_none:
        return None
    if len(args) == 0:
        return None
    if len(args) == 1:
        return args[0]
    return args


class _Y(object):
    def __init__(
        self,
        prefix=nv,
        output=nv,
        serialize=nv,
        show_line_number=nv,
        show_time=nv,
        show_delta=nv,
        show_enter=nv,
        show_exit=nv,
        show_traceback=nv,
        sort_dicts=nv,
        enabled=nv,
        line_length=nv,
        compact=nv,
        indent=nv,
        depth=nv,
        wrap_indent=nv,
        context_separator=nv,
        separator=nv,
        equals_separator=nv,
        values_only=nv,
        values_only_for_fstrings=nv,
        return_none=nv,
        enforce_line_length=nv,
        decorator=nv,
        context_manager=nv,
        delta=nv,
        _parent=nv,
        **kwargs
    ):
        self._attributes = {}
        if _parent is nv:
            self._parent = default
        else:
            self._parent = _parent
        for key in vars(default):
            setattr(self, key, None)

        if _parent == default:
            func = "y.new()"
        else:
            func = "y.fork()"
        self.assign(kwargs, locals(), func=func)

        self.check()

    def __repr__(self):
        pairs = []
        for key in vars(default):
            if not key.startswith("__"):
                value = getattr(self, key)
                if not callable(value):
                    pairs.append(str(key) + "=" + repr(value))
        return "y.new(" + ", ".join(pairs) + ")"

    def __getattr__(self, item):
        if item in shortcut_to_name:
            item = shortcut_to_name[item]
        if item == "delta":
            return perf_counter() - getattr(self, "start_time")

        if item in self._attributes:
            if self._attributes[item] is None:
                return getattr(self._parent, item)
            else:
                return self._attributes[item]
        raise AttributeError("{item} not found".format(item=item))

    def __setattr__(self, item, value):
        if item in shortcut_to_name:
            item = shortcut_to_name[item]
        if item == "delta":
            item = "start_time"
            if value is not None:
                value = perf_counter() - value

        if item in ["_attributes"]:
            super(_Y, self).__setattr__(item, value)
        else:
            self._attributes[item] = value

    def assign(self, shortcuts, source, func):
        for key, value in shortcuts.items():
            if key in shortcut_to_name:
                if value is not nv:
                    full_name = shortcut_to_name[key]
                    if source[full_name] is nv:
                        source[full_name] = value
                    else:
                        raise ValueError("can't use {key} and {full_name} in {func}".format(key=key, full_name=full_name, func=func))
            else:
                raise TypeError("{func} got an unexpected keyword argument {key}".format(func=func, key=key))
        for key, value in source.items():
            if value is not nv:
                if key == "delta":
                    key = "start_time"
                    if value is not None:
                        value = perf_counter() - value
                setattr(self, key, value)

    def fork(self, **kwargs):
        kwargs["_parent"] = self
        return _Y(**kwargs)

    def __call__(self, *args, **kwargs):
        prefix = kwargs.pop("prefix", nv)
        output = kwargs.pop("output", nv)
        serialize = kwargs.pop("serialize", nv)
        show_line_number = kwargs.pop("show_line_number", nv)
        show_time = kwargs.pop("show_time", nv)
        show_delta = kwargs.pop("show_delta", nv)
        show_enter = kwargs.pop("show_enter", nv)
        show_exit = kwargs.pop("show_exit", nv)
        show_traceback = kwargs.pop("show_traceback", nv)
        sort_dicts = kwargs.pop("sort_dicts", nv)
        enabled = kwargs.pop("enabled", nv)
        line_length = kwargs.pop("line_length", nv)
        compact = kwargs.pop("compact", nv)
        indent = kwargs.pop("indent", nv)
        depth = kwargs.pop("depth", nv)
        wrap_indent = kwargs.pop("wrap_indent", nv)
        context_separator = kwargs.pop("context_separator", nv)
        separator = kwargs.pop("separator", nv)
        equals_separator = kwargs.pop("equals_separator", nv)
        values_only = kwargs.pop("values_only", nv)
        values_only_for_fstrings = kwargs.pop("values_only_for_fstrings", nv)
        return_none = kwargs.pop("return_none", nv)
        enforce_line_length = kwargs.pop("enforce_line_length", nv)
        decorator = kwargs.pop("decorator", nv)
        context_manager = kwargs.pop("context_manager", nv)
        delta = kwargs.pop("delta", nv)
        as_str = kwargs.pop("as_str", nv)
        as_str = False if as_str is nv else bool(as_str)

        self.is_context_manager = False

        Pair = collections.namedtuple("Pair", "left right")

        this = self.fork()
        this.assign(kwargs, locals(), func="__call__")

        if this.enabled == [] and not (as_str or this.decorator or this.context_manager):
            return return_args(args, this.return_none)

        this.check()

        call_frame = inspect.currentframe()
        filename0 = call_frame.f_code.co_filename

        call_frame = call_frame.f_back
        filename = call_frame.f_code.co_filename

        if filename == filename0:
            call_frame = call_frame.f_back
            filename = call_frame.f_code.co_filename

        if filename in ("<stdin>", "<string>"):
            filename_name = ""
            code = "\n\n"
            this_line = ""
            this_line_prev = ""
            line_number = 0
            parent_function = ""
        else:
            try:
                main_file = sys.modules["__main__"].__file__
                main_file_resolved = os.path.abspath(main_file)
            except AttributeError:
                main_file_resolved = None
            filename_resolved = os.path.abspath(filename)
            if (filename.startswith("<") and filename.endswith(">")) or (main_file_resolved is None) or (filename_resolved == main_file_resolved):
                filename_name = ""
            else:
                filename_name = "[" + os.path.basename(filename) + "]"

            if filename not in codes:
                frame_info = inspect.getframeinfo(call_frame, context=1000000)  # get the full source code
                if frame_info.code_context is None:
                    no_source_error()
                codes[filename] = frame_info.code_context
            code = codes[filename]
            frame_info = inspect.getframeinfo(call_frame, context=1)
            parent_function = frame_info.function
            if parent_function == "<module>":
                parent_function = ""
            else:
                parent_function = " in {parent_function}()".format(parent_function=parent_function)
            line_number = frame_info.lineno
            if 0 <= line_number - 1 < len(code):
                this_line = code[line_number - 1].strip()
            else:
                this_line = ""
            if 0 <= line_number - 2 < len(code):
                this_line_prev = code[line_number - 2].strip()
            else:
                this_line_prev = ""
        if this_line.startswith("@") or this_line_prev.startswith("@") or this.decorator:
            if as_str:
                raise TypeError("as_str may not be True when y used as decorator")

            for ln, line in enumerate(code[line_number - 1 :], line_number):
                if line.strip().startswith("def") or line.strip().startswith("class"):
                    line_number = ln
                    break
            else:
                line_number += 1
            this.line_number_with_filename_and_parent = "#{line_number}{filename_name}{parent_function}".format(
                line_number=line_number, filename_name=filename_name, parent_function=parent_function
            )

            def real_decorator(function):
                @functools.wraps(function)
                def wrapper(*args, **kwargs):
                    enter_time = perf_counter()
                    context = this.context()

                    args_kwargs = [repr(arg) for arg in args] + ["{k}={repr_v}".format(k=k, repr_v=repr(v)) for k, v in kwargs.items()]
                    function_arguments = function.__name__ + "(" + (", ".join(args_kwargs)) + ")"

                    if this.show_enter:
                        this.do_output(
                            "{context}called {function_arguments}{traceback}".format(
                                context=context, function_arguments=function_arguments, traceback=this.traceback()
                            )
                        )
                    result = function(*args, **kwargs)
                    duration = perf_counter() - enter_time

                    context = this.context()
                    if this.show_exit:
                        this.do_output(
                            "{context}returned {repr_result} from {function_arguments} in {duration:.6f} seconds{traceback}".format(
                                context=context, repr_result=repr(result), function_arguments=function_arguments, duration=duration, traceback=this.traceback()
                            )
                        )

                    return result

                return wrapper

            if len(args) == 0:
                return real_decorator

            if len(args) == 1 and callable(args[0]):
                return real_decorator(args[0])
            raise TypeError("arguments are not allowed in y used as decorator")

        if filename in ("<stdin>", "<string>"):
            this.line_number_with_filename_and_parent = ""
        else:
            call_node = Source.executing(call_frame).node
            if call_node is None:
                no_source_error()
            line_number = call_node.lineno
            this_line = code[line_number - 1].strip()

            this.line_number_with_filename_and_parent = "#{line_number}{filename_name}{parent_function}".format(
                line_number=line_number, filename_name=filename_name, parent_function=parent_function
            )

        if this_line.startswith("with ") or this_line.startswith("with\t") or this.context_manager:
            if as_str:
                raise TypeError("as_str may not be True when y used as context manager")
            if args:
                raise TypeError("non-keyword arguments are not allowed when y used as context manager")

            this.is_context_manager = True
            return this

        if not this.enabled and not as_str:
            return return_args(args, this.return_none)

        if args:
            context = this.context()
            pairs = []
            if filename in ("<stdin>", "<string>") or this.values_only:
                for right in args:
                    pairs.append(Pair(left="", right=right))
            else:
                source = Source.for_frame(call_frame)
                for node, right in zip(call_node.args, args):
                    left = source.asttokens().get_text(node)
                    if "\n" in left:
                        left = " " * node.first_token.start[1] + left
                        left = textwrap.dedent(left)
                    try:
                        ast.literal_eval(left)  # it's indeed a literal
                        left = ""
                    except Exception:
                        pass
                    if left:
                        try:
                            if isinstance(left, str):
                                s = ast.parse(left, mode='eval')
                            if isinstance(s, ast.Expression):
                                s = s.body
                            if s and isinstance(s, ast.JoinedStr): # it is indeed an f-string
                                if this.values_only_for_fstrings:
                                    left = ""  
                        except Exception:
                            pass
                    if left:
                        left += this.equals_separator
                    pairs.append(Pair(left=left, right=right))

            just_one_line = False
            if not (len(pairs) > 1 and this.separator == ""):
                if not any("\n" in pair.left for pair in pairs):
                    as_one_line = context + this.separator.join(pair.left + this.serialize_kwargs(obj=pair.right, width=10000) for pair in pairs)
                    if len(as_one_line) <= this.line_length and "\n" not in as_one_line:
                        out = as_one_line
                        just_one_line = True

            if not just_one_line:
                if isinstance(this.wrap_indent, numbers.Number):
                    wrap_indent = int(this.wrap_indent) * " "
                else:
                    wrap_indent = str(this.wrap_indent)

                if context.strip():
                    if len(context.rstrip()) >= len(wrap_indent):
                        indent1 = wrap_indent
                        indent1_rest = wrap_indent
                        lines = [context]
                    else:
                        indent1 = context.rstrip().ljust(len(wrap_indent))
                        indent1_rest = wrap_indent
                        lines = []
                else:
                    indent1 = ""
                    indent1_rest = ""
                    lines = []

                for pair in pairs:
                    do_right = False
                    if "\n" in pair.left:
                        for s in pair.left.splitlines():
                            lines.append(indent1 + s)
                            do_right = True
                    else:
                        start = indent1 + pair.left
                        line = start + this.serialize_kwargs(obj=pair.right, width=this.line_length - len(start))
                        if "\n" in line:
                            lines.append(start)
                            do_right = True
                        else:
                            lines.append(line)
                    indent1 = indent1_rest
                    if do_right:
                        indent2 = indent1 + wrap_indent
                        line = this.serialize_kwargs(obj=pair.right, width=this.line_length - len(indent2))
                        for s in line.splitlines():
                            lines.append(indent2 + s)

                out = "\n".join(line.rstrip() for line in lines)

        else:
            this.show_line_number = True
            out = this.context(omit_context_separator=True)

        if this.show_traceback:
            out += this.traceback()

        if as_str:
            if this.enforce_line_length:
                out = "\n".join(line[: this.line_length] for line in out.splitlines())
            return out + "\n"
        this.do_output(out)

        return return_args(args, this.return_none)

    def configure(
        self,
        prefix=nv,
        output=nv,
        serialize=nv,
        show_line_number=nv,
        show_time=nv,
        show_delta=nv,
        show_enter=nv,
        show_exit=nv,
        show_traceback=nv,
        sort_dicts=nv,
        enabled=nv,
        line_length=nv,
        compact=nv,
        indent=nv,
        depth=nv,
        wrap_indent=nv,
        context_separator=nv,
        separator=nv,
        equals_separator=nv,
        values_only=nv,
        values_only_for_fstrings=nv,
        return_none=nv,
        enforce_line_length=nv,
        decorator=nv,
        context_manager=nv,
        delta=nv,
        **kwargs
    ):
        self.assign(kwargs, locals(), "configure()")
        self.check()
        return self

    def new(self, ignore_json=False, **kwargs):
        if ignore_json:
            return _Y(_parent=default_pre_json, **kwargs)
        else:
            return _Y(**kwargs)

    def clone(
        self,
        prefix=nv,
        output=nv,
        serialize=nv,
        show_line_number=nv,
        show_time=nv,
        show_delta=nv,
        show_enter=nv,
        show_exit=nv,
        show_traceback=nv,
        sort_dicts=nv,
        enabled=nv,
        line_length=nv,
        compact=nv,
        indent=nv,
        depth=nv,
        wrap_indent=nv,
        context_separator=nv,
        separator=nv,
        equals_separator=nv,
        values_only=nv,
        values_only_for_fstrings=nv,
        return_none=nv,
        enforce_line_length=nv,
        decorator=nv,
        context_manager=nv,
        delta=nv,
        **kwargs
    ):
        this = _Y(_parent=self._parent)
        this.assign({}, self._attributes, func="clone()")
        this.assign(kwargs, locals(), func="clone()")

        return this

    @contextlib.contextmanager
    def preserve(self):
        save = dict(self._attributes)
        yield
        self._attributes = save

    def __enter__(self):
        if not hasattr(self, "is_context_manager"):
            raise ValueError("not allowed as context_manager")
        self.save_traceback = self.traceback()
        self.enter_time = perf_counter()
        if self.show_enter:
            context = self.context()
            self.do_output(context + "enter" + self.save_traceback)
        return self

    def __exit__(self, *args):
        if self.show_exit:
            context = self.context()
            duration = perf_counter() - self.enter_time
            self.do_output("{context}exit in {duration:.6f} seconds{traceback}".format(context=context, duration=duration, traceback=self.save_traceback))
        self.is_context_manager = False

    def context(self, omit_context_separator=False):
        if self.show_line_number and self.line_number_with_filename_and_parent != "":
            parts = [self.line_number_with_filename_and_parent]
        else:
            parts = []
        if self.show_time:
            parts.append("@ " + str(datetime.datetime.now().strftime("%H:%M:%S.%f")))

        if self.show_delta:
            t0 = perf_counter() - self.start_time
            parts.append("delta={t0:.3f}".format(t0=t0))

        context = " ".join(parts)
        if not omit_context_separator and context:
            context += self.context_separator

        return (self.prefix() if callable(self.prefix) else self.prefix) + context

    def do_output(self, s):
        if self.enforce_line_length:
            s = "\n".join(line[: self.line_length] for line in s.splitlines())
        if self.enabled:
            if callable(self.output):
                self.output(s)
            elif self.output == "stderr":
                print(s, file=sys.stderr)
            elif self.output == "stdout":
                print(s, file=sys.stdout)
            elif self.output == "logging.debug":
                logging.debug(s)
            elif self.output == "logging.info":
                logging.info(s)
            elif self.output == "logging.warning":
                logging.warning(s)
            elif self.output == "logging.error":
                logging.error(s)
            elif self.output == "logging.critical":
                logging.critical(s)
            elif self.output in ("", "null"):
                pass

            elif isinstance(self.output, str):
                if PY2:
                    with io.open(self.output, "a+", encoding="utf-8") as f:
                        print(s, file=f)
                if PY3:
                    with open(self.output, "a+", encoding="utf-8") as f:
                        print(s, file=f)
            elif isinstance(self.output, Path):
                with self.output.open("a+", encoding="utf-8") as f:
                    print(s, file=f)

            else:
                print(s, file=self.output)

    def traceback(self):
        if self.show_traceback:
            if isinstance(self.wrap_indent, numbers.Number):
                wrap_indent = int(self.wrap_indent) * " "
            else:
                wrap_indent = str(self.wrap_indent)

            result = "\n" + wrap_indent + "Traceback (most recent call last)\n"
            #  Python 2.7 does not allow entry.filename, entry.line, etc, so we have to index entry
            return result + "\n".join(
                wrap_indent + '  File "' + entry[0] + '", line ' + str(entry[1]) + ", in " + entry[2] + "\n" + wrap_indent + "    " + entry[3]
                for entry in traceback.extract_stack()[:-2]
            )
        else:
            return ""

    def check(self):

        if self.decorator and self.context_manager:
            raise TypeError("decorator and context_manager can't be specified both.")

        if callable(self.output):
            return
        if isinstance(self.output, (str, Path)):
            return
        try:
            if PY2:
                self.output.write(unicode(""))
            if PY3:
                self.output.write("")
            return

        except Exception:
            pass
        raise TypeError("output should be a callable, str, Path or open text file.")

    if PY2:

        def serialize_kwargs(self, obj, width):

            kwargs = {key: getattr(self, key) for key in ("indent", "depth") if key in inspect.getargspec(self.serialize).args}
            kwargs["width"] = width

            return self.serialize(obj, **kwargs)

    if PY3:

        def serialize_kwargs(self, obj, width):
            kwargs = {key: getattr(self, key) for key in ("sort_dicts", "compact", "indent", "depth") if key in inspect.signature(self.serialize).parameters}
            if "width" in inspect.signature(self.serialize).parameters:
                kwargs["width"] = width
            return self.serialize(obj, **kwargs)


codes = {}

set_defaults()
default_pre_json = copy.copy(default)
apply_json()
y = _Y()
yc = y.fork(prefix="yc| ")

# source of asttoke.util

# Copyright 2016 Grist Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import collections
import token


def token_repr(tok_type, string):
    """Returns a human-friendly representation of a token with the given type and string."""
    # repr() prefixes unicode with 'u' on Python2 but not Python3; strip it out for consistency.
    return "%s:%s" % (token.tok_name[tok_type], repr(string).lstrip("u"))


class Token(collections.namedtuple("Token", "type string start end line index startpos endpos")):
    """
  TokenInfo is an 8-tuple containing the same 5 fields as the tokens produced by the tokenize
  module, and 3 additional ones useful for this module:

  - [0] .type     Token type (see token.py)
  - [1] .string   Token (a string)
  - [2] .start    Starting (row, column) indices of the token (a 2-tuple of ints)
  - [3] .end      Ending (row, column) indices of the token (a 2-tuple of ints)
  - [4] .line     Original line (string)
  - [5] .index    Index of the token in the list of tokens that it belongs to.
  - [6] .startpos Starting character offset into the input text.
  - [7] .endpos   Ending character offset into the input text.
  """

    def __str__(self):
        return token_repr(self.type, self.string)


def match_token(token, tok_type, tok_str=None):
    """Returns true if token is of the given type and, if a string is given, has that string."""
    return token.type == tok_type and (tok_str is None or token.string == tok_str)


def expect_token(token, tok_type, tok_str=None):
    """
  Verifies that the given token is of the expected type. If tok_str is given, the token string
  is verified too. If the token doesn't match, raises an informative ValueError.
  """
    if not match_token(token, tok_type, tok_str):
        raise ValueError("Expected token %s, got %s on line %s col %s" % (token_repr(tok_type, tok_str), str(token), token.start[0], token.start[1] + 1))


# These were previously defined in tokenize.py and distinguishable by being greater than
# token.N_TOKEN. As of python3.7, they are in token.py, and we check for them explicitly.
if hasattr(token, "ENCODING"):

    def is_non_coding_token(token_type):
        """
    These are considered non-coding tokens, as they don't affect the syntax tree.
    """
        return token_type in (token.NL, token.COMMENT, token.ENCODING)


else:

    def is_non_coding_token(token_type):
        """
    These are considered non-coding tokens, as they don't affect the syntax tree.
    """
        return token_type >= token.N_TOKENS


def iter_children_func(node):
    """
  Returns a function which yields all direct children of a AST node,
  skipping children that are singleton nodes.
  The function depends on whether ``node`` is from ``ast`` or from the ``astroid`` module.
  """
    return iter_children_astroid if hasattr(node, "get_children") else iter_children_ast


def iter_children_astroid(node):
    # Don't attempt to process children of JoinedStr nodes, which we can't fully handle yet.
    if is_joined_str(node):
        return []

    return node.get_children()


SINGLETONS = {
    c for n, c in iteritems(ast.__dict__) if isinstance(c, type) and issubclass(c, (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop))
}


def iter_children_ast(node):
    # Don't attempt to process children of JoinedStr nodes, which we can't fully handle yet.
    if is_joined_str(node):
        return

    if isinstance(node, ast.Dict):
        # override the iteration order: instead of <all keys>, <all values>,
        # yield keys and values in source order (key1, value1, key2, value2, ...)
        for (key, value) in zip(node.keys, node.values):
            if key is not None:
                yield key
            yield value
        return

    for child in ast.iter_child_nodes(node):
        # Skip singleton children; they don't reflect particular positions in the code and break the
        # assumptions about the tree consisting of distinct nodes. Note that collecting classes
        # beforehand and checking them in a set is faster than using isinstance each time.
        if child.__class__ not in SINGLETONS:
            yield child


stmt_class_names = {n for n, c in iteritems(ast.__dict__) if isinstance(c, type) and issubclass(c, ast.stmt)}
expr_class_names = {n for n, c in iteritems(ast.__dict__) if isinstance(c, type) and issubclass(c, ast.expr)} | {
    "AssignName",
    "DelName",
    "Const",
    "AssignAttr",
    "DelAttr",
}

# These feel hacky compared to isinstance() but allow us to work with both ast and astroid nodes
# in the same way, and without even importing astroid.
def is_expr(node):
    """Returns whether node is an expression node."""
    return node.__class__.__name__ in expr_class_names


def is_stmt(node):
    """Returns whether node is a statement node."""
    return node.__class__.__name__ in stmt_class_names


def is_module(node):
    """Returns whether node is a module node."""
    return node.__class__.__name__ == "Module"


def is_joined_str(node):
    """Returns whether node is a JoinedStr node, used to represent f-strings."""
    # At the moment, nodes below JoinedStr have wrong line/col info, and trying to process them only
    # leads to errors.
    return node.__class__.__name__ == "JoinedStr"


def is_slice(node):
    """Returns whether node represents a slice, e.g. `1:2` in `x[1:2]`"""
    # Before 3.9, a tuple containing a slice is an ExtSlice,
    # but this was removed in https://bugs.python.org/issue34822
    return node.__class__.__name__ in ("Slice", "ExtSlice") or (node.__class__.__name__ == "Tuple" and any(map(is_slice, node.elts)))


# Sentinel value used by visit_tree().
_PREVISIT = object()


def visit_tree(node, previsit, postvisit):
    """
  Scans the tree under the node depth-first using an explicit stack. It avoids implicit recursion
  via the function call stack to avoid hitting 'maximum recursion depth exceeded' error.

  It calls ``previsit()`` and ``postvisit()`` as follows:

  * ``previsit(node, par_value)`` - should return ``(par_value, value)``
        ``par_value`` is as returned from ``previsit()`` of the parent.

  * ``postvisit(node, par_value, value)`` - should return ``value``
        ``par_value`` is as returned from ``previsit()`` of the parent, and ``value`` is as
        returned from ``previsit()`` of this node itself. The return ``value`` is ignored except
        the one for the root node, which is returned from the overall ``visit_tree()`` call.

  For the initial node, ``par_value`` is None. ``postvisit`` may be None.
  """
    if not postvisit:
        postvisit = lambda node, pvalue, value: None

    iter_children = iter_children_func(node)
    done = set()
    ret = None
    stack = [(node, None, _PREVISIT)]
    while stack:
        current, par_value, value = stack.pop()
        if value is _PREVISIT:
            assert current not in done  # protect againt infinite loop in case of a bad tree.
            done.add(current)

            pvalue, post_value = previsit(current, par_value)
            stack.append((current, par_value, post_value))

            # Insert all children in reverse order (so that first child ends up on top of the stack).
            ins = len(stack)
            for n in iter_children(current):
                stack.insert(ins, (n, pvalue, _PREVISIT))
        else:
            ret = postvisit(current, par_value, value)
    return ret


def walk(node):
    """
  Recursively yield all descendant nodes in the tree starting at ``node`` (including ``node``
  itself), using depth-first pre-order traversal (yieling parents before their children).

  This is similar to ``ast.walk()``, but with a different order, and it works for both ``ast`` and
  ``astroid`` trees. Also, as ``iter_children()``, it skips singleton nodes generated by ``ast``.
  """
    iter_children = iter_children_func(node)
    done = set()
    stack = [node]
    while stack:
        current = stack.pop()
        assert current not in done  # protect againt infinite loop in case of a bad tree.
        done.add(current)

        yield current

        # Insert all children in reverse order (so that first child ends up on top of the stack).
        # This is faster than building a list and reversing it.
        ins = len(stack)
        for c in iter_children(current):
            stack.insert(ins, c)


def replace(text, replacements):
    """
  Replaces multiple slices of text with new values. This is a convenience method for making code
  modifications of ranges e.g. as identified by ``ASTTokens.get_text_range(node)``. Replacements is
  an iterable of ``(start, end, new_text)`` tuples.

  For example, ``replace("this is a test", [(0, 4, "X"), (8, 9, "THE")])`` produces
  ``"X is THE test"``.
  """
    p = 0
    parts = []
    for (start, end, new_text) in sorted(replacements):
        parts.append(text[p:start])
        parts.append(new_text)
        p = end
    parts.append(text[p:])
    return "".join(parts)


class NodeMethods(object):
    """
  Helper to get `visit_{node_type}` methods given a node's class and cache the results.
  """

    def __init__(self):
        self._cache = {}

    def get(self, obj, cls):
        """
    Using the lowercase name of the class as node_type, returns `obj.visit_{node_type}`,
    or `obj.visit_default` if the type-specific method is not found.
    """
        method = self._cache.get(cls)
        if not method:
            name = "visit_" + cls.__name__.lower()
            method = getattr(obj, name, obj.visit_default)
            self._cache[cls] = method
        return method


# end of source of asttoke.util

# source of asttokens.mark_tokens

# Copyright 2016 Grist Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import sys
import token

# Mapping of matching braces. To find a token here, look up token[:2].
_matching_pairs_left = {(token.OP, "("): (token.OP, ")"), (token.OP, "["): (token.OP, "]"), (token.OP, "{"): (token.OP, "}")}

_matching_pairs_right = {(token.OP, ")"): (token.OP, "("), (token.OP, "]"): (token.OP, "["), (token.OP, "}"): (token.OP, "{")}


class MarkTokens(object):
    """
  Helper that visits all nodes in the AST tree and assigns .first_token and .last_token attributes
  to each of them. This is the heart of the token-marking logic.
  """

    def __init__(self, code):
        self._code = code
        self._methods = NodeMethods()
        self._iter_children = None

    def visit_tree(self, node):
        self._iter_children = iter_children_func(node)
        visit_tree(node, self._visit_before_children, self._visit_after_children)

    def _visit_before_children(self, node, parent_token):
        col = getattr(node, "col_offset", None)
        token = self._code.get_token_from_utf8(node.lineno, col) if col is not None else None

        if not token and is_module(node):
            # We'll assume that a Module node starts at the start of the source code.
            token = self._code.get_token(1, 0)

        # Use our own token, or our parent's if we don't have one, to pass to child calls as
        # parent_token argument. The second value becomes the token argument of _visit_after_children.
        return (token or parent_token, token)

    def _visit_after_children(self, node, parent_token, token):
        # This processes the node generically first, after all children have been processed.

        # Get the first and last tokens that belong to children. Note how this doesn't assume that we
        # iterate through children in order that corresponds to occurrence in source code. This
        # assumption can fail (e.g. with return annotations).
        first = token
        last = None
        for child in self._iter_children(node):
            if not first or child.first_token.index < first.index:
                first = child.first_token
            if not last or child.last_token.index > last.index:
                last = child.last_token

        # If we don't have a first token from _visit_before_children, and there were no children, then
        # use the parent's token as the first token.
        first = first or parent_token

        # If no children, set last token to the first one.
        last = last or first

        # Statements continue to before NEWLINE. This helps cover a few different cases at once.
        if is_stmt(node):
            last = self._find_last_in_stmt(last)

        # Capture any unmatched brackets.
        first, last = self._expand_to_matching_pairs(first, last, node)

        # Give a chance to node-specific methods to adjust.
        nfirst, nlast = self._methods.get(self, node.__class__)(node, first, last)

        if (nfirst, nlast) != (first, last):
            # If anything changed, expand again to capture any unmatched brackets.
            nfirst, nlast = self._expand_to_matching_pairs(nfirst, nlast, node)

        node.first_token = nfirst
        node.last_token = nlast

    def _find_last_in_stmt(self, start_token):
        t = start_token
        while not match_token(t, token.NEWLINE) and not match_token(t, token.OP, ";") and not token.ISEOF(t.type):
            t = self._code.next_token(t, include_extra=True)
        return self._code.prev_token(t)

    def _expand_to_matching_pairs(self, first_token, last_token, node):
        """
    Scan tokens in [first_token, last_token] range that are between node's children, and for any
    unmatched brackets, adjust first/last tokens to include the closing pair.
    """
        # We look for opening parens/braces among non-child tokens (i.e. tokens between our actual
        # child nodes). If we find any closing ones, we match them to the opens.
        to_match_right = []
        to_match_left = []
        for tok in self._code.token_range(first_token, last_token):
            tok_info = tok[:2]
            if to_match_right and tok_info == to_match_right[-1]:
                to_match_right.pop()
            elif tok_info in _matching_pairs_left:
                to_match_right.append(_matching_pairs_left[tok_info])
            elif tok_info in _matching_pairs_right:
                to_match_left.append(_matching_pairs_right[tok_info])

        # Once done, extend `last_token` to match any unclosed parens/braces.
        for match in reversed(to_match_right):
            last = self._code.next_token(last_token)
            # Allow for trailing commas or colons (allowed in subscripts) before the closing delimiter
            while any(match_token(last, token.OP, x) for x in (",", ":")):
                last = self._code.next_token(last)
            # Now check for the actual closing delimiter.
            if match_token(last, *match):
                last_token = last

        # And extend `first_token` to match any unclosed opening parens/braces.
        for match in to_match_left:
            first = self._code.prev_token(first_token)
            if match_token(first, *match):
                first_token = first

        return (first_token, last_token)

    # ----------------------------------------------------------------------
    # Node visitors. Each takes a preliminary first and last tokens, and returns the adjusted pair
    # that will actually be assigned.

    def visit_default(self, node, first_token, last_token):
        # pylint: disable=no-self-use
        # By default, we don't need to adjust the token we computed earlier.
        return (first_token, last_token)

    def handle_comp(self, open_brace, node, first_token, last_token):
        # For list/set/dict comprehensions, we only get the token of the first child, so adjust it to
        # include the opening brace (the closing brace will be matched automatically).
        before = self._code.prev_token(first_token)
        expect_token(before, token.OP, open_brace)
        return (before, last_token)

    # Python 3.8 fixed the starting position of list comprehensions:
    # https://bugs.python.org/issue31241
    if sys.version_info < (3, 8):

        def visit_listcomp(self, node, first_token, last_token):
            return self.handle_comp("[", node, first_token, last_token)

    if PY2:
        # We shouldn't do this on PY3 because its SetComp/DictComp already have a correct start.
        def visit_setcomp(self, node, first_token, last_token):
            return self.handle_comp("{", node, first_token, last_token)

        def visit_dictcomp(self, node, first_token, last_token):
            return self.handle_comp("{", node, first_token, last_token)

    def visit_comprehension(self, node, first_token, last_token):
        # The 'comprehension' node starts with 'for' but we only get first child; we search backwards
        # to find the 'for' keyword.
        first = self._code.find_token(first_token, token.NAME, "for", reverse=True)
        return (first, last_token)

    def visit_if(self, node, first_token, last_token):
        while first_token.string not in ("if", "elif"):
            first_token = self._code.prev_token(first_token)
        return first_token, last_token

    def handle_attr(self, node, first_token, last_token):
        # Attribute node has ".attr" (2 tokens) after the last child.
        dot = self._code.find_token(last_token, token.OP, ".")
        name = self._code.next_token(dot)
        expect_token(name, token.NAME)
        return (first_token, name)

    visit_attribute = handle_attr
    visit_assignattr = handle_attr
    visit_delattr = handle_attr

    def handle_def(self, node, first_token, last_token):
        # With astroid, nodes that start with a doc-string can have an empty body, in which case we
        # need to adjust the last token to include the doc string.
        if not node.body and getattr(node, "doc", None):
            last_token = self._code.find_token(last_token, token.STRING)

        # Include @ from decorator
        if first_token.index > 0:
            prev = self._code.prev_token(first_token)
            if match_token(prev, token.OP, "@"):
                first_token = prev
        return (first_token, last_token)

    visit_classdef = handle_def
    visit_functiondef = handle_def

    def handle_following_brackets(self, node, last_token, opening_bracket):
        # This is for calls and subscripts, which have a pair of brackets
        # at the end which may contain no nodes, e.g. foo() or bar[:].
        # We look for the opening bracket and then let the matching pair be found automatically
        # Remember that last_token is at the end of all children,
        # so we are not worried about encountering a bracket that belongs to a child.
        first_child = next(self._iter_children(node))
        call_start = self._code.find_token(first_child.last_token, token.OP, opening_bracket)
        if call_start.index > last_token.index:
            last_token = call_start
        return last_token

    def visit_call(self, node, first_token, last_token):
        last_token = self.handle_following_brackets(node, last_token, "(")

        # Handling a python bug with decorators with empty parens, e.g.
        # @deco()
        # def ...
        if match_token(first_token, token.OP, "@"):
            first_token = self._code.next_token(first_token)
        return (first_token, last_token)

    def visit_subscript(self, node, first_token, last_token):
        last_token = self.handle_following_brackets(node, last_token, "[")
        return (first_token, last_token)

    def handle_bare_tuple(self, node, first_token, last_token):
        # A bare tuple doesn't include parens; if there is a trailing comma, make it part of the tuple.
        maybe_comma = self._code.next_token(last_token)
        if match_token(maybe_comma, token.OP, ","):
            last_token = maybe_comma
        return (first_token, last_token)

    if sys.version_info >= (3, 8):
        # In Python3.8 parsed tuples include parentheses when present.
        def handle_tuple_nonempty(self, node, first_token, last_token):
            # It's a bare tuple if the first token belongs to the first child. The first child may
            # include extraneous parentheses (which don't create new nodes), so account for those too.
            child = node.elts[0]
            child_first, child_last = self._gobble_parens(child.first_token, child.last_token, True)
            if first_token == child_first:
                return self.handle_bare_tuple(node, first_token, last_token)
            return (first_token, last_token)

    else:
        # Before python 3.8, parsed tuples do not include parens.
        def handle_tuple_nonempty(self, node, first_token, last_token):
            (first_token, last_token) = self.handle_bare_tuple(node, first_token, last_token)
            return self._gobble_parens(first_token, last_token, False)

    def visit_tuple(self, node, first_token, last_token):
        if not node.elts:
            # An empty tuple is just "()", and we need no further info.
            return (first_token, last_token)
        return self.handle_tuple_nonempty(node, first_token, last_token)

    def _gobble_parens(self, first_token, last_token, include_all=False):
        # Expands a range of tokens to include one or all pairs of surrounding parentheses, and
        # returns (first, last) tokens that include these parens.
        while first_token.index > 0:
            prev = self._code.prev_token(first_token)
            next = self._code.next_token(last_token)
            if match_token(prev, token.OP, "(") and match_token(next, token.OP, ")"):
                first_token, last_token = prev, next
                if include_all:
                    continue
            break
        return (first_token, last_token)

    def visit_str(self, node, first_token, last_token):
        return self.handle_str(first_token, last_token)

    def visit_joinedstr(self, node, first_token, last_token):
        return self.handle_str(first_token, last_token)

    def visit_bytes(self, node, first_token, last_token):
        return self.handle_str(first_token, last_token)

    def handle_str(self, first_token, last_token):
        # Multiple adjacent STRING tokens form a single string.
        last = self._code.next_token(last_token)
        while match_token(last, token.STRING):
            last_token = last
            last = self._code.next_token(last_token)
        return (first_token, last_token)

    def handle_num(self, node, value, first_token, last_token):
        # A constant like '-1' gets turned into two tokens; this will skip the '-'.
        while match_token(last_token, token.OP):
            last_token = self._code.next_token(last_token)

        if isinstance(value, complex):
            # A complex number like -2j cannot be compared directly to 0
            # A complex number like 1-2j is expressed as a binary operation
            # so we don't need to worry about it
            value = value.imag

        # This makes sure that the - is included
        if value < 0 and first_token.type == token.NUMBER:
            first_token = self._code.prev_token(first_token)
        return (first_token, last_token)

    def visit_num(self, node, first_token, last_token):
        return self.handle_num(node, node.n, first_token, last_token)

    # In Astroid, the Num and Str nodes are replaced by Const.
    def visit_const(self, node, first_token, last_token):
        if isinstance(node.value, numbers.Number):
            return self.handle_num(node, node.value, first_token, last_token)
        elif isinstance(node.value, (text_type, binary_type)):
            return self.visit_str(node, first_token, last_token)
        return (first_token, last_token)

    # In Python >= 3.6, there is a similar class 'Constant' for literals
    # In 3.8 it became the type produced by ast.parse
    # https://bugs.python.org/issue32892
    visit_constant = visit_const

    def visit_keyword(self, node, first_token, last_token):
        # Until python 3.9 (https://bugs.python.org/issue40141),
        # ast.keyword nodes didn't have line info. Astroid has lineno None.
        if node.arg is not None and getattr(node, "lineno", None) is None:
            equals = self._code.find_token(first_token, token.OP, "=", reverse=True)
            name = self._code.prev_token(equals)
            expect_token(name, token.NAME, node.arg)
            first_token = name
        return (first_token, last_token)

    def visit_starred(self, node, first_token, last_token):
        # Astroid has 'Starred' nodes (for "foo(*bar)" type args), but they need to be adjusted.
        if not match_token(first_token, token.OP, "*"):
            star = self._code.prev_token(first_token)
            if match_token(star, token.OP, "*"):
                first_token = star
        return (first_token, last_token)

    def visit_assignname(self, node, first_token, last_token):
        # Astroid may turn 'except' clause into AssignName, but we need to adjust it.
        if match_token(first_token, token.NAME, "except"):
            colon = self._code.find_token(last_token, token.OP, ":")
            first_token = last_token = self._code.prev_token(colon)
        return (first_token, last_token)

    if PY2:
        # No need for this on Python3, which already handles 'with' nodes correctly.
        def visit_with(self, node, first_token, last_token):
            first = self._code.find_token(first_token, token.NAME, "with", reverse=True)
            return (first, last_token)

    # Async nodes should typically start with the word 'async'
    # but Python < 3.7 doesn't put the col_offset there
    # AsyncFunctionDef is slightly different because it might have
    # decorators before that, which visit_functiondef handles
    def handle_async(self, node, first_token, last_token):
        if not first_token.string == "async":
            first_token = self._code.prev_token(first_token)
        return (first_token, last_token)

    visit_asyncfor = handle_async
    visit_asyncwith = handle_async

    def visit_asyncfunctiondef(self, node, first_token, last_token):
        if match_token(first_token, token.NAME, "def"):
            # Include the 'async' token
            first_token = self._code.prev_token(first_token)
        return self.visit_functiondef(node, first_token, last_token)


# end of source of asttokens.mark_tokens

# source of asttokens.line_numbers

# Copyright 2016 Grist Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import re

_line_start_re = re.compile(r"^", re.M)


class LineNumbers(object):
    """
  Class to convert between character offsets in a text string, and pairs (line, column) of 1-based
  line and 0-based column numbers, as used by tokens and AST nodes.

  This class expects unicode for input and stores positions in unicode. But it supports
  translating to and from utf8 offsets, which are used by ast parsing.
  """

    def __init__(self, text):
        # A list of character offsets of each line's first character.
        self._line_offsets = [m.start(0) for m in _line_start_re.finditer(text)]
        self._text = text
        self._text_len = len(text)
        self._utf8_offset_cache = {}  # maps line num to list of char offset for each byte in line

    def from_utf8_col(self, line, utf8_column):
        """
    Given a 1-based line number and 0-based utf8 column, returns a 0-based unicode column.
    """
        offsets = self._utf8_offset_cache.get(line)
        if offsets is None:
            end_offset = self._line_offsets[line] if line < len(self._line_offsets) else self._text_len
            line_text = self._text[self._line_offsets[line - 1] : end_offset]

            offsets = [i for i, c in enumerate(line_text) for byte in c.encode("utf8")]
            offsets.append(len(line_text))
            self._utf8_offset_cache[line] = offsets

        return offsets[max(0, min(len(offsets) - 1, utf8_column))]

    def line_to_offset(self, line, column):
        """
    Converts 1-based line number and 0-based column to 0-based character offset into text.
    """
        line -= 1
        if line >= len(self._line_offsets):
            return self._text_len
        elif line < 0:
            return 0
        else:
            return min(self._line_offsets[line] + max(0, column), self._text_len)

    def offset_to_line(self, offset):
        """
    Converts 0-based character offset to pair (line, col) of 1-based line and 0-based column
    numbers.
    """
        offset = max(0, min(self._text_len, offset))
        line_index = bisect.bisect_right(self._line_offsets, offset) - 1
        return (line_index + 1, offset - self._line_offsets[line_index])


# end of source of asttokens.line_numbers

# source of asttokens.asttokens

# Copyright 2016 Grist Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import bisect
import token
import tokenize
import io


class ASTTokens(object):
    """
  ASTTokens maintains the text of Python code in several forms: as a string, as line numbers, and
  as tokens, and is used to mark and access token and position information.

  ``source_text`` must be a unicode or UTF8-encoded string. If you pass in UTF8 bytes, remember
  that all offsets you'll get are to the unicode text, which is available as the ``.text``
  property.

  If ``parse`` is set, the ``source_text`` will be parsed with ``ast.parse()``, and the resulting
  tree marked with token info and made available as the ``.tree`` property.

  If ``tree`` is given, it will be marked and made available as the ``.tree`` property. In
  addition to the trees produced by the ``ast`` module, ASTTokens will also mark trees produced
  using ``astroid`` library <https://www.astroid.org>.

  If only ``source_text`` is given, you may use ``.mark_tokens(tree)`` to mark the nodes of an AST
  tree created separately.
  """

    def __init__(self, source_text, parse=False, tree=None, filename="<unknown>"):
        self._filename = filename
        self._tree = ast.parse(source_text, filename) if parse else tree

        # Decode source after parsing to let Python 2 handle coding declarations.
        # (If the encoding was not utf-8 compatible, then even if it parses correctly,
        # we'll fail with a unicode error here.)
        if isinstance(source_text, binary_type):
            source_text = source_text.decode("utf8")

        self._text = source_text
        self._line_numbers = LineNumbers(source_text)

        # Tokenize the code.
        self._tokens = list(self._generate_tokens(source_text))

        # Extract the start positions of all tokens, so that we can quickly map positions to tokens.
        self._token_offsets = [tok.startpos for tok in self._tokens]

        if self._tree:
            self.mark_tokens(self._tree)

    def mark_tokens(self, root_node):
        """
    Given the root of the AST or Astroid tree produced from source_text, visits all nodes marking
    them with token and position information by adding ``.first_token`` and
    ``.last_token``attributes. This is done automatically in the constructor when ``parse`` or
    ``tree`` arguments are set, but may be used manually with a separate AST or Astroid tree.
    """
        # The hard work of this class is done by MarkTokens
        MarkTokens(self).visit_tree(root_node)

    def _generate_tokens(self, text):
        """
    Generates tokens for the given code.
    """
        # This is technically an undocumented API for Python3, but allows us to use the same API as for
        # Python2. See http://stackoverflow.com/a/4952291/328565.
        for index, tok in enumerate(tokenize.generate_tokens(io.StringIO(text).readline)):
            tok_type, tok_str, start, end, line = tok
            yield Token(
                tok_type,
                tok_str,
                start,
                end,
                line,
                index,
                self._line_numbers.line_to_offset(start[0], start[1]),
                self._line_numbers.line_to_offset(end[0], end[1]),
            )

    @property
    def text(self):
        """The source code passed into the constructor."""
        return self._text

    @property
    def tokens(self):
        """The list of tokens corresponding to the source code from the constructor."""
        return self._tokens

    @property
    def tree(self):
        """The root of the AST tree passed into the constructor or parsed from the source code."""
        return self._tree

    @property
    def filename(self):
        """The filename that was parsed"""
        return self._filename

    def get_token_from_offset(self, offset):
        """
    Returns the token containing the given character offset (0-based position in source text),
    or the preceeding token if the position is between tokens.
    """
        return self._tokens[bisect.bisect(self._token_offsets, offset) - 1]

    def get_token(self, lineno, col_offset):
        """
    Returns the token containing the given (lineno, col_offset) position, or the preceeding token
    if the position is between tokens.
    """
        # TODO: add test for multibyte unicode. We need to translate offsets from ast module (which
        # are in utf8) to offsets into the unicode text. tokenize module seems to use unicode offsets
        # but isn't explicit.
        return self.get_token_from_offset(self._line_numbers.line_to_offset(lineno, col_offset))

    def get_token_from_utf8(self, lineno, col_offset):
        """
    Same as get_token(), but interprets col_offset as a UTF8 offset, which is what `ast` uses.
    """
        return self.get_token(lineno, self._line_numbers.from_utf8_col(lineno, col_offset))

    def next_token(self, tok, include_extra=False):
        """
    Returns the next token after the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
        i = tok.index + 1
        if not include_extra:
            while is_non_coding_token(self._tokens[i].type):
                i += 1
        return self._tokens[i]

    def prev_token(self, tok, include_extra=False):
        """
    Returns the previous token before the given one. If include_extra is True, includes non-coding
    tokens from the tokenize module, such as NL and COMMENT.
    """
        i = tok.index - 1
        if not include_extra:
            while is_non_coding_token(self._tokens[i].type):
                i -= 1
        return self._tokens[i]

    def find_token(self, start_token, tok_type, tok_str=None, reverse=False):
        """
    Looks for the first token, starting at start_token, that matches tok_type and, if given, the
    token string. Searches backwards if reverse is True. Returns ENDMARKER token if not found (you
    can check it with `token.ISEOF(t.type)`.
    """
        t = start_token
        advance = self.prev_token if reverse else self.next_token
        while not match_token(t, tok_type, tok_str) and not token.ISEOF(t.type):
            t = advance(t, include_extra=True)
        return t

    def token_range(self, first_token, last_token, include_extra=False):
        """
    Yields all tokens in order from first_token through and including last_token. If
    include_extra is True, includes non-coding tokens such as tokenize.NL and .COMMENT.
    """
        for i in xrange(first_token.index, last_token.index + 1):
            if include_extra or not is_non_coding_token(self._tokens[i].type):
                yield self._tokens[i]

    def get_tokens(self, node, include_extra=False):
        """
    Yields all tokens making up the given node. If include_extra is True, includes non-coding
    tokens such as tokenize.NL and .COMMENT.
    """
        return self.token_range(node.first_token, node.last_token, include_extra=include_extra)

    def get_text_range(self, node):
        """
    After mark_tokens() has been called, returns the (startpos, endpos) positions in source text
    corresponding to the given node. Returns (0, 0) for nodes (like `Load`) that don't correspond
    to any particular text.
    """
        if not hasattr(node, "first_token"):
            return (0, 0)

        start = node.first_token.startpos
        if any(match_token(t, token.NEWLINE) for t in self.get_tokens(node)):
            # Multi-line nodes would be invalid unless we keep the indentation of the first node.
            start = self._text.rfind("\n", 0, start) + 1

        return (start, node.last_token.endpos)

    def get_text(self, node):
        """
    After mark_tokens() has been called, returns the text corresponding to the given node. Returns
    '' for nodes (like `Load`) that don't correspond to any particular text.
    """
        start, end = self.get_text_range(node)
        return self._text[start:end]


# end of source of asttokens.asttokens

# source of executing

import __future__
import ast
import dis
import functools
import inspect
import io
import linecache
import sys
import types
from collections import defaultdict, namedtuple
from itertools import islice
from operator import attrgetter
from threading import RLock

if PY3:
    # noinspection PyUnresolvedReferences
    from functools import lru_cache

    # noinspection PyUnresolvedReferences
    from tokenize import detect_encoding
    from itertools import zip_longest

    # noinspection PyUnresolvedReferences,PyCompatibility
    from pathlib import Path

    cache = lru_cache(maxsize=None)
    text_type = str
else:
    from lib2to3.pgen2.tokenize import detect_encoding, cookie_re as encoding_pattern
    from itertools import izip_longest as zip_longest

    #    class Path(object):
    #        pass

    def cache(func):
        d = {}

        @functools.wraps(func)
        def wrapper(*args):
            if args in d:
                return d[args]
            result = d[args] = func(*args)
            return result

        return wrapper

    # noinspection PyUnresolvedReferences
    text_type = unicode
try:
    # noinspection PyUnresolvedReferences
    _get_instructions = dis.get_instructions
except AttributeError:

    class Instruction(namedtuple("Instruction", "offset argval opname starts_line")):
        lineno = None

    from dis import HAVE_ARGUMENT, EXTENDED_ARG, hasconst, opname, findlinestarts

    # Based on dis.disassemble from 2.7
    # Left as similar as possible for easy diff

    def _get_instructions(co):
        code = co.co_code
        linestarts = dict(findlinestarts(co))
        n = len(code)
        i = 0
        extended_arg = 0
        while i < n:
            offset = i
            c = code[i]
            op = ord(c)
            lineno = linestarts.get(i)
            argval = None
            i = i + 1
            if op >= HAVE_ARGUMENT:
                oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
                extended_arg = 0
                i = i + 2
                if op == EXTENDED_ARG:
                    extended_arg = oparg * 65536

                if op in hasconst:
                    argval = co.co_consts[oparg]
            yield Instruction(offset, argval, opname[op], lineno)


def assert_(condition, message=""):
    """
    Like an assert statement, but unaffected by -O
    :param condition: value that is expected to be truthy
    :type message: Any
    """
    if not condition:
        raise AssertionError(str(message))


def get_instructions(co):
    lineno = None
    for inst in _get_instructions(co):
        lineno = inst.starts_line or lineno
        assert_(lineno)
        inst.lineno = lineno
        yield inst


TESTING = 0


class NotOneValueFound(Exception):
    pass


def only(it):
    if hasattr(it, "__len__"):
        if len(it) != 1:
            raise NotOneValueFound("Expected one value, found %s" % len(it))
        # noinspection PyTypeChecker
        return list(it)[0]

    lst = tuple(islice(it, 2))
    if len(lst) == 0:
        raise NotOneValueFound("Expected one value, found 0")
    if len(lst) > 1:
        raise NotOneValueFound("Expected one value, found several")
    return lst[0]


class Source(object):
    """
    The source code of a single file and associated metadata.

    The main method of interest is the classmethod `executing(frame)`.

    If you want an instance of this class, don't construct it.
    Ideally use the classmethod `for_frame(frame)`.
    If you don't have a frame, use `for_filename(filename [, module_globals])`.
    These methods cache instances by filename, so at most one instance exists per filename.

    Attributes:
        - filename
        - text
        - lines
        - tree: AST parsed from text, or None if text is not valid Python
            All nodes in the tree have an extra `parent` attribute

    Other methods of interest:
        - statements_at_line
        - asttokens
        - code_qualname
    """

    def __init__(self, filename, lines):
        """
        Don't call this constructor, see the class docstring.
        """

        self.filename = filename
        text = "".join(lines)

        if not isinstance(text, text_type):
            encoding = self.detect_encoding(text)
            # noinspection PyUnresolvedReferences
            text = text.decode(encoding)
            lines = [line.decode(encoding) for line in lines]

        self.text = text
        self.lines = [line.rstrip("\r\n") for line in lines]

        if PY3:
            ast_text = text
        else:
            # In python 2 it's a syntax error to parse unicode
            # with an encoding declaration, so we remove it but
            # leave empty lines in its place to keep line numbers the same
            ast_text = "".join(["\n" if i < 2 and encoding_pattern.match(line) else line for i, line in enumerate(lines)])

        self._nodes_by_line = defaultdict(list)
        self.tree = None
        self._qualnames = {}

        try:
            self.tree = ast.parse(ast_text, filename=filename)
        except SyntaxError:
            pass
        else:
            for node in ast.walk(self.tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node
                if hasattr(node, "lineno"):
                    self._nodes_by_line[node.lineno].append(node)

            visitor = QualnameVisitor()
            visitor.visit(self.tree)
            self._qualnames = visitor.qualnames

    @classmethod
    def for_frame(cls, frame, use_cache=True):
        """
        Returns the `Source` object corresponding to the file the frame is executing in.
        """
        return cls.for_filename(frame.f_code.co_filename, frame.f_globals or {}, use_cache)

    @classmethod
    def for_filename(cls, filename, module_globals=None, use_cache=True):
        if isinstance(filename, Path):
            filename = str(filename)

        source_cache = cls._class_local("__source_cache", {})
        if use_cache:
            try:
                return source_cache[filename]
            except KeyError:
                pass

        if not use_cache:
            linecache.checkcache(filename)

        lines = tuple(linecache.getlines(filename, module_globals))
        result = source_cache[filename] = cls._for_filename_and_lines(filename, lines)
        return result

    @classmethod
    def _for_filename_and_lines(cls, filename, lines):
        source_cache = cls._class_local("__source_cache_with_lines", {})
        try:
            return source_cache[(filename, lines)]
        except KeyError:
            pass

        result = source_cache[(filename, lines)] = cls(filename, lines)
        return result

    @classmethod
    def lazycache(cls, frame):
        if hasattr(linecache, "lazycache"):
            linecache.lazycache(frame.f_code.co_filename, frame.f_globals)

    @classmethod
    def executing(cls, frame_or_tb):
        """
        Returns an `Executing` object representing the operation
        currently executing in the given frame or traceback object.
        """
        if isinstance(frame_or_tb, types.TracebackType):
            # https://docs.python.org/3/reference/datamodel.html#traceback-objects
            # "tb_lineno gives the line number where the exception occurred;
            #  tb_lasti indicates the precise instruction.
            #  The line number and last instruction in the traceback may differ
            #  from the line number of its frame object
            #  if the exception occurred in a try statement with no matching except clause
            #  or with a finally clause."
            tb = frame_or_tb
            frame = tb.tb_frame
            lineno = tb.tb_lineno
            lasti = tb.tb_lasti
        else:
            frame = frame_or_tb
            lineno = frame.f_lineno
            lasti = frame.f_lasti

        code = frame.f_code
        key = (code, id(code), lasti)
        executing_cache = cls._class_local("__executing_cache", {})

        try:
            args = executing_cache[key]
        except KeyError:

            def find(source, retry_cache):
                node = stmts = None
                tree = source.tree
                if tree:
                    try:
                        stmts = source.statements_at_line(lineno)
                        if stmts:
                            if code.co_filename.startswith("<ipython-input-") and code.co_name == "<module>":
                                tree = _extract_ipython_statement(stmts, tree)
                            node = NodeFinder(frame, stmts, tree, lasti).result
                    except Exception as e:
                        # These exceptions can be caused by the source code having changed
                        # so the cached Source doesn't match the running code
                        # (e.g. when using IPython %autoreload)
                        # Try again with a fresh Source object
                        if retry_cache and isinstance(e, (NotOneValueFound, AssertionError)):
                            return find(source=cls.for_frame(frame, use_cache=False), retry_cache=False)
                        if TESTING:
                            raise

                    if node:
                        new_stmts = {statement_containing_node(node)}
                        assert_(new_stmts <= stmts)
                        stmts = new_stmts

                return source, node, stmts

            args = find(source=cls.for_frame(frame), retry_cache=True)
            executing_cache[key] = args

        return Executing(frame, *args)

    @classmethod
    def _class_local(cls, name, default):
        """
        Returns an attribute directly associated with this class
        (as opposed to subclasses), setting default if necessary
        """
        # classes have a mappingproxy preventing us from using setdefault
        result = cls.__dict__.get(name, default)
        setattr(cls, name, result)
        return result

    @cache
    def statements_at_line(self, lineno):
        """
        Returns the statement nodes overlapping the given line.

        Returns at most one statement unless semicolons are present.

        If the `text` attribute is not valid python, meaning
        `tree` is None, returns an empty set.

        Otherwise, `Source.for_frame(frame).statements_at_line(frame.f_lineno)`
        should return at least one statement.
        """

        return {statement_containing_node(node) for node in self._nodes_by_line[lineno]}

    @cache
    def asttokens(self):
        """
        Returns an ASTTokens object for getting the source of specific AST nodes.

        See http://asttokens.readthedocs.io/en/latest/api-index.html
        """
        return ASTTokens(self.text, tree=self.tree, filename=self.filename)

    @staticmethod
    def decode_source(source):
        if isinstance(source, bytes):
            encoding = Source.detect_encoding(source)
            source = source.decode(encoding)
        return source

    @staticmethod
    def detect_encoding(source):
        return detect_encoding(io.BytesIO(source).readline)[0]

    def code_qualname(self, code):
        """
        Imitates the __qualname__ attribute of functions for code objects.
        Given:

            - A function `func`
            - A frame `frame` for an execution of `func`, meaning:
                `frame.f_code is func.__code__`

        `Source.for_frame(frame).code_qualname(frame.f_code)`
        will be equal to `func.__qualname__`*. Works for Python 2 as well,
        where of course no `__qualname__` attribute exists.

        Falls back to `code.co_name` if there is no appropriate qualname.

        Based on https://github.com/wbolster/qualname

        (* unless `func` is a lambda
        nested inside another lambda on the same line, in which case
        the outer lambda's qualname will be returned for the codes
        of both lambdas)
        """
        assert_(code.co_filename == self.filename)
        return self._qualnames.get((code.co_name, code.co_firstlineno), code.co_name)


class Executing(object):
    """
    Information about the operation a frame is currently executing.

    Generally you will just want `node`, which is the AST node being executed,
    or None if it's unknown.
    """

    def __init__(self, frame, source, node, stmts):
        self.frame = frame
        self.source = source
        self.node = node
        self.statements = stmts

    def code_qualname(self):
        return self.source.code_qualname(self.frame.f_code)

    def text(self):
        return self.source.asttokens().get_text(self.node)

    def text_range(self):
        return self.source.asttokens().get_text_range(self.node)


class QualnameVisitor(ast.NodeVisitor):
    def __init__(self):
        super(QualnameVisitor, self).__init__()
        self.stack = []
        self.qualnames = {}

    def add_qualname(self, node, name=None):
        name = name or node.name
        self.stack.append(name)
        if getattr(node, "decorator_list", ()):
            lineno = node.decorator_list[0].lineno
        else:
            lineno = node.lineno
        self.qualnames.setdefault((name, lineno), ".".join(self.stack))

    def visit_FunctionDef(self, node, name=None):
        self.add_qualname(node, name)
        self.stack.append("<locals>")
        if isinstance(node, ast.Lambda):
            children = [node.body]
        else:
            children = node.body
        for child in children:
            self.visit(child)
        self.stack.pop()
        self.stack.pop()

        # Find lambdas in the function definition outside the body,
        # e.g. decorators or default arguments
        # Based on iter_child_nodes
        for field, child in ast.iter_fields(node):
            if field == "body":
                continue
            if isinstance(child, ast.AST):
                self.visit(child)
            elif isinstance(child, list):
                for grandchild in child:
                    if isinstance(grandchild, ast.AST):
                        self.visit(grandchild)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        # noinspection PyTypeChecker
        self.visit_FunctionDef(node, "<lambda>")

    def visit_ClassDef(self, node):
        self.add_qualname(node)
        self.generic_visit(node)
        self.stack.pop()


future_flags = sum(getattr(__future__, fname).compiler_flag for fname in __future__.all_feature_names)


def compile_similar_to(source, matching_code):
    return compile(source, matching_code.co_filename, "exec", flags=future_flags & matching_code.co_flags, dont_inherit=True)


sentinel = "io8urthglkjdghvljusketgIYRFYUVGHFRTBGVHKGF78678957647698"


class NodeFinder(object):
    def __init__(self, frame, stmts, tree, lasti):
        self.frame = frame
        self.tree = tree
        self.code = code = frame.f_code
        self.is_pytest = any("pytest" in name.lower() for group in [code.co_names, code.co_varnames] for name in group)

        if self.is_pytest:
            self.ignore_linenos = frozenset(assert_linenos(tree))
        else:
            self.ignore_linenos = frozenset()

        instruction = self.get_actual_current_instruction(lasti)
        op_name = instruction.opname
        self.lasti = instruction.offset

        if op_name.startswith("CALL_"):
            typ = ast.Call
        elif op_name.startswith(("BINARY_SUBSCR", "SLICE+")):
            typ = ast.Subscript
        elif op_name.startswith("BINARY_"):
            typ = ast.BinOp
        elif op_name.startswith("UNARY_"):
            typ = ast.UnaryOp
        elif op_name in ("LOAD_ATTR", "LOAD_METHOD", "LOOKUP_METHOD"):
            typ = ast.Attribute
        elif op_name in ("COMPARE_OP", "IS_OP", "CONTAINS_OP"):
            typ = ast.Compare
        else:
            raise RuntimeError(op_name)

        with lock:
            exprs = {
                node for stmt in stmts for node in ast.walk(stmt) if isinstance(node, typ) if not (hasattr(node, "ctx") and not isinstance(node.ctx, ast.Load))
            }

            self.result = only(list(self.matching_nodes(exprs)))

    def clean_instructions(self, code):
        return [inst for inst in get_instructions(code) if inst.opname != "EXTENDED_ARG" if inst.lineno not in self.ignore_linenos]

    def get_original_clean_instructions(self):
        result = self.clean_instructions(self.code)

        # pypy sometimes (when is not clear)
        # inserts JUMP_IF_NOT_DEBUG instructions in bytecode
        # If they're not present in our compiled instructions,
        # ignore them in the original bytecode
        if not any(inst.opname == "JUMP_IF_NOT_DEBUG" for inst in self.compile_instructions()):
            result = [inst for inst in result if inst.opname != "JUMP_IF_NOT_DEBUG"]

        return result

    def matching_nodes(self, exprs):
        original_instructions = self.get_original_clean_instructions()
        original_index = only(i for i, inst in enumerate(original_instructions) if inst.offset == self.lasti)
        for i, expr in enumerate(exprs):
            setter = get_setter(expr)
            # noinspection PyArgumentList
            replacement = ast.BinOp(left=expr, op=ast.Pow(), right=ast.Str(s=sentinel))
            ast.fix_missing_locations(replacement)
            setter(replacement)
            try:
                instructions = self.compile_instructions()
            finally:
                setter(expr)
            indices = [i for i, instruction in enumerate(instructions) if instruction.argval == sentinel]

            # There can be several indices when the bytecode is duplicated,
            # as happens in a finally block in 3.9+
            # First we remove the opcodes caused by our modifications
            for index_num, sentinel_index in enumerate(indices):
                # Adjustment for removing sentinel instructions below
                # in past iterations
                sentinel_index -= index_num * 2

                assert_(instructions.pop(sentinel_index).opname == "LOAD_CONST")
                assert_(instructions.pop(sentinel_index).opname == "BINARY_POWER")

            # Then we see if any of the instruction indices match
            for index_num, sentinel_index in enumerate(indices):
                sentinel_index -= index_num * 2
                new_index = sentinel_index - 1

                if new_index != original_index:
                    continue

                original_inst = original_instructions[original_index]
                new_inst = instructions[new_index]

                # In Python 3.9+, changing 'not x in y' to 'not sentinel_transformation(x in y)'
                # changes a CONTAINS_OP(invert=1) to CONTAINS_OP(invert=0),<sentinel stuff>,UNARY_NOT
                if (
                    original_inst.opname == new_inst.opname in ("CONTAINS_OP", "IS_OP")
                    and original_inst.arg != new_inst.arg
                    and (original_instructions[original_index + 1].opname != instructions[new_index + 1].opname == "UNARY_NOT")
                ):
                    # Remove the difference for the upcoming assert
                    instructions.pop(new_index + 1)

                # Check that the modified instructions don't have anything unexpected
                for inst1, inst2 in zip_longest(original_instructions, instructions):
                    assert_(
                        inst1.opname == inst2.opname
                        or all("JUMP_IF_" in inst.opname for inst in [inst1, inst2])
                        or all(inst.opname in ("JUMP_FORWARD", "JUMP_ABSOLUTE") for inst in [inst1, inst2])
                        or (inst1.opname == "PRINT_EXPR" and inst2.opname == "POP_TOP")
                        or (inst1.opname in ("LOAD_METHOD", "LOOKUP_METHOD") and inst2.opname == "LOAD_ATTR")
                        or (inst1.opname == "CALL_METHOD" and inst2.opname == "CALL_FUNCTION"),
                        (inst1, inst2, ast.dump(expr), expr.lineno, self.code.co_filename),
                    )

                yield expr

    def compile_instructions(self):
        module_code = compile_similar_to(self.tree, self.code)
        code = only(self.find_codes(module_code))
        return self.clean_instructions(code)

    def find_codes(self, root_code):
        checks = [attrgetter("co_firstlineno"), attrgetter("co_name"), attrgetter("co_freevars"), attrgetter("co_cellvars")]
        if not self.is_pytest:
            checks += [attrgetter("co_names"), attrgetter("co_varnames")]

        def matches(c):
            return all(f(c) == f(self.code) for f in checks)

        code_options = []
        if matches(root_code):
            code_options.append(root_code)

        def finder(code):
            for const in code.co_consts:
                if not inspect.iscode(const):
                    continue

                if matches(const):
                    code_options.append(const)
                finder(const)

        finder(root_code)
        return code_options

    def get_actual_current_instruction(self, lasti):
        """
        Get the instruction corresponding to the current
        frame offset, skipping EXTENDED_ARG instructions
        """
        # Don't use get_original_clean_instructions
        # because we need the actual instructions including
        # EXTENDED_ARG
        instructions = list(get_instructions(self.code))
        index = only(i for i, inst in enumerate(instructions) if inst.offset == lasti)

        while True:
            instruction = instructions[index]
            if instruction.opname != "EXTENDED_ARG":
                return instruction
            index += 1


def get_setter(node):
    parent = node.parent
    for name, field in ast.iter_fields(parent):
        if field is node:
            return lambda new_node: setattr(parent, name, new_node)
        elif isinstance(field, list):
            for i, item in enumerate(field):
                if item is node:

                    def setter(new_node):
                        field[i] = new_node

                    return setter


lock = RLock()


@cache
def statement_containing_node(node):
    while not isinstance(node, ast.stmt):
        node = node.parent
    return node


def assert_linenos(tree):
    for node in ast.walk(tree):
        if hasattr(node, "parent") and hasattr(node, "lineno") and isinstance(statement_containing_node(node), ast.Assert):
            yield node.lineno


def _extract_ipython_statement(stmts, tree):
    # IPython separates each statement in a cell to be executed separately
    # So NodeFinder should only compile one statement at a time or it
    # will find a code mismatch.
    stmt = list(stmts)[0]
    while not isinstance(stmt.parent, ast.Module):
        stmt = stmt.parent
    # use `ast.parse` instead of `ast.Module` for better portability
    # python3.8 changes the signature of `ast.Module`
    # Inspired by https://github.com/pallets/werkzeug/pull/1552/files
    tree = ast.parse("")
    tree.body = [stmt]
    ast.copy_location(tree, stmt)
    return tree


# end of soource of executing


# source of pprint (3.8) module

#
#  Author:      Fred L. Drake, Jr.
#               fdrake@acm.org
#
#  This is a simple little module I wrote to make life easier.  I didn't
#  see anything quite like it in the library, though I may have overlooked
#  something.  I wrote this when I was trying to read some heavily nested
#  tuples with fairly non-descriptive content.  This is modeled very much
#  after Lisp/Scheme - style pretty-printing of lists.  If you find it
#  useful, thank small children who sleep at night.

"""Support to pretty-print lists, tuples, & dictionaries recursively.

Very simple, but useful, especially in debugging data structures.

Classes
-------

PrettyPrinter()
    Handle pretty-printing operations onto a stream using a configured
    set of formatting parameters.

Functions
---------

pformat()
    Format a Python object into a pretty-printed representation.

pprint()
    Pretty-print a Python object to a stream [default is sys.stdout].

saferepr()
    Generate a 'standard' repr()-like value, but protect against recursive
    data structures.

"""

import collections as _collections
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO


def pprint(object, stream=None, indent=1, width=80, depth=None, compact=False, sort_dicts=True):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(stream=stream, indent=indent, width=width, depth=depth, compact=compact, sort_dicts=sort_dicts)
    printer.pprint(object)


def pformat(object, indent=1, width=80, depth=None, compact=False, sort_dicts=True):
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(indent=indent, width=width, depth=depth, compact=compact, sort_dicts=sort_dicts).pformat(object)


def pp(object, *args, **kwargs):
    sort_dicts = kwargs.pop("sort_dicts", False)
    """Pretty-print a Python object"""
    pprint(object, *args, sort_dicts=sort_dicts, **kwargs)


def saferepr(object):
    """Version of repr() which can handle recursive data structures."""
    return _safe_repr(object, {}, None, 0, True)[0]


def isreadable(object):
    """Determine if saferepr(object) is readable by eval()."""
    return _safe_repr(object, {}, None, 0, True)[1]


def isrecursive(object):
    """Determine if object requires a recursive representation."""
    return _safe_repr(object, {}, None, 0, True)[2]


class _safe_key:
    """Helper function for key functions when sorting unorderable objects.

    The wrapped-object will fallback to a Py2.x style comparison for
    unorderable types (sorting first comparing the type name and then by
    the obj ids).  Does not work recursively, so dict.items() must have
    _safe_key applied to both the key and the value.

    """

    __slots__ = ["obj"]

    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        try:
            return self.obj < other.obj
        except TypeError:
            return (str(type(self.obj)), id(self.obj)) < (str(type(other.obj)), id(other.obj))


def _safe_tuple(t):
    "Helper function for comparing 2-tuples"
    return _safe_key(t[0]), _safe_key(t[1])


if PY3:

    class PrettyPrinter:
        def __init__(self, indent=1, width=80, depth=None, stream=None, compact=False, sort_dicts=True):
            """Handle pretty printing operations onto a stream using a set of
        configured parameters.

        indent
            Number of spaces to indent for each level of nesting.

        width
            Attempted maximum number of columns in the output.

        depth
            The maximum depth to print out nested structures.

        stream
            The desired output stream.  If omitted (or false), the standard
            output stream available at construction will be used.

        compact
            If true, several items will be combined in one line.

        sort_dicts
            If true, dict keys are sorted.

        """
            indent = int(indent)
            width = int(width)
            if indent < 0:
                raise ValueError("indent must be >= 0")
            if depth is not None and depth <= 0:
                raise ValueError("depth must be > 0")
            if not width:
                raise ValueError("width must be != 0")
            self._depth = depth
            self._indent_per_level = indent
            self._width = width
            if stream is not None:
                self._stream = stream
            else:
                self._stream = _sys.stdout
            self._compact = bool(compact)
            self._sort_dicts = sort_dicts

        def pprint(self, object):
            self._format(object, self._stream, 0, 0, {}, 0)
            self._stream.write("\n")

        def pformat(self, object):
            sio = _StringIO()
            self._format(object, sio, 0, 0, {}, 0)
            return sio.getvalue()

        def isrecursive(self, object):
            return self.format(object, {}, 0, 0)[2]

        def isreadable(self, object):
            s, readable, recursive = self.format(object, {}, 0, 0)
            return readable and not recursive

        def _format(self, object, stream, indent, allowance, context, level):
            objid = id(object)
            if objid in context:
                stream.write(_recursion(object))
                self._recursive = True
                self._readable = False
                return
            rep = self._repr(object, context, level)
            max_width = self._width - indent - allowance
            if len(rep) > max_width:
                p = self._dispatch.get(type(object).__repr__, None)
                if p is not None:
                    context[objid] = 1
                    p(self, object, stream, indent, allowance, context, level + 1)
                    del context[objid]
                    return
                elif isinstance(object, dict):
                    context[objid] = 1
                    self._pprint_dict(object, stream, indent, allowance, context, level + 1)
                    del context[objid]
                    return
            stream.write(rep)

        _dispatch = {}

        def _pprint_dict(self, object, stream, indent, allowance, context, level):
            write = stream.write
            write("{")
            if self._indent_per_level > 1:
                write((self._indent_per_level - 1) * " ")
            length = len(object)
            if length:
                if self._sort_dicts:
                    items = sorted(object.items(), key=_safe_tuple)
                else:
                    items = object.items()
                self._format_dict_items(items, stream, indent, allowance + 1, context, level)
            write("}")

        _dispatch[dict.__repr__] = _pprint_dict

        def _pprint_ordered_dict(self, object, stream, indent, allowance, context, level):
            if not len(object):
                stream.write(repr(object))
                return
            cls = object.__class__
            stream.write(cls.__name__ + "(")
            self._format(list(object.items()), stream, indent + len(cls.__name__) + 1, allowance + 1, context, level)
            stream.write(")")

        _dispatch[_collections.OrderedDict.__repr__] = _pprint_ordered_dict

        def _pprint_list(self, object, stream, indent, allowance, context, level):
            stream.write("[")
            self._format_items(object, stream, indent, allowance + 1, context, level)
            stream.write("]")

        _dispatch[list.__repr__] = _pprint_list

        def _pprint_tuple(self, object, stream, indent, allowance, context, level):
            stream.write("(")
            endchar = ",)" if len(object) == 1 else ")"
            self._format_items(object, stream, indent, allowance + len(endchar), context, level)
            stream.write(endchar)

        _dispatch[tuple.__repr__] = _pprint_tuple

        def _pprint_set(self, object, stream, indent, allowance, context, level):
            if not len(object):
                stream.write(repr(object))
                return
            typ = object.__class__
            if typ is set:
                stream.write("{")
                endchar = "}"
            else:
                stream.write(typ.__name__ + "({")
                endchar = "})"
                indent += len(typ.__name__) + 1
            object = sorted(object, key=_safe_key)
            self._format_items(object, stream, indent, allowance + len(endchar), context, level)
            stream.write(endchar)

        _dispatch[set.__repr__] = _pprint_set
        _dispatch[frozenset.__repr__] = _pprint_set

        def _pprint_str(self, object, stream, indent, allowance, context, level):
            write = stream.write
            if not len(object):
                write(repr(object))
                return
            chunks = []
            lines = object.splitlines(True)
            if level == 1:
                indent += 1
                allowance += 1
            max_width1 = max_width = self._width - indent
            for i, line in enumerate(lines):
                rep = repr(line)
                if i == len(lines) - 1:
                    max_width1 -= allowance
                if len(rep) <= max_width1:
                    chunks.append(rep)
                else:
                    # A list of alternating (non-space, space) strings
                    parts = re.findall(r"\S*\s*", line)
                    assert parts
                    assert not parts[-1]
                    parts.pop()  # drop empty last part
                    max_width2 = max_width
                    current = ""
                    for j, part in enumerate(parts):
                        candidate = current + part
                        if j == len(parts) - 1 and i == len(lines) - 1:
                            max_width2 -= allowance
                        if len(repr(candidate)) > max_width2:
                            if current:
                                chunks.append(repr(current))
                            current = part
                        else:
                            current = candidate
                    if current:
                        chunks.append(repr(current))
            if len(chunks) == 1:
                write(rep)
                return
            if level == 1:
                write("(")
            for i, rep in enumerate(chunks):
                if i > 0:
                    write("\n" + " " * indent)
                write(rep)
            if level == 1:
                write(")")

        _dispatch[str.__repr__] = _pprint_str

        def _pprint_bytes(self, object, stream, indent, allowance, context, level):
            write = stream.write
            if len(object) <= 4:
                write(repr(object))
                return
            parens = level == 1
            if parens:
                indent += 1
                allowance += 1
                write("(")
            delim = ""
            for rep in _wrap_bytes_repr(object, self._width - indent, allowance):
                write(delim)
                write(rep)
                if not delim:
                    delim = "\n" + " " * indent
            if parens:
                write(")")

        _dispatch[bytes.__repr__] = _pprint_bytes

        def _pprint_bytearray(self, object, stream, indent, allowance, context, level):
            write = stream.write
            write("bytearray(")
            self._pprint_bytes(bytes(object), stream, indent + 10, allowance + 1, context, level + 1)
            write(")")

        _dispatch[bytearray.__repr__] = _pprint_bytearray

        def _pprint_mappingproxy(self, object, stream, indent, allowance, context, level):
            stream.write("mappingproxy(")
            self._format(object.copy(), stream, indent + 13, allowance + 1, context, level)
            stream.write(")")

        _dispatch[_types.MappingProxyType.__repr__] = _pprint_mappingproxy

        def _format_dict_items(self, items, stream, indent, allowance, context, level):
            write = stream.write
            indent += self._indent_per_level
            delimnl = ",\n" + " " * indent
            last_index = len(items) - 1
            for i, (key, ent) in enumerate(items):
                last = i == last_index
                rep = self._repr(key, context, level)
                write(rep)
                write(": ")
                self._format(ent, stream, indent + len(rep) + 2, allowance if last else 1, context, level)
                if not last:
                    write(delimnl)

        def _format_items(self, items, stream, indent, allowance, context, level):
            write = stream.write
            indent += self._indent_per_level
            if self._indent_per_level > 1:
                write((self._indent_per_level - 1) * " ")
            delimnl = ",\n" + " " * indent
            delim = ""
            width = max_width = self._width - indent + 1
            it = iter(items)
            try:
                next_ent = next(it)
            except StopIteration:
                return
            last = False
            while not last:
                ent = next_ent
                try:
                    next_ent = next(it)
                except StopIteration:
                    last = True
                    max_width -= allowance
                    width -= allowance
                if self._compact:
                    rep = self._repr(ent, context, level)
                    w = len(rep) + 2
                    if width < w:
                        width = max_width
                        if delim:
                            delim = delimnl
                    if width >= w:
                        width -= w
                        write(delim)
                        delim = ", "
                        write(rep)
                        continue
                write(delim)
                delim = delimnl
                self._format(ent, stream, indent, allowance if last else 1, context, level)

        def _repr(self, object, context, level):
            repr, readable, recursive = self.format(object, context.copy(), self._depth, level)
            if not readable:
                self._readable = False
            if recursive:
                self._recursive = True
            return repr

        def format(self, object, context, maxlevels, level):
            """Format object for a specific context, returning a string
        and flags indicating whether the representation is 'readable'
        and whether the object represents a recursive construct.
        """
            return _safe_repr(object, context, maxlevels, level, self._sort_dicts)

        def _pprint_default_dict(self, object, stream, indent, allowance, context, level):
            if not len(object):
                stream.write(repr(object))
                return
            rdf = self._repr(object.default_factory, context, level)
            cls = object.__class__
            indent += len(cls.__name__) + 1
            stream.write("%s(%s,\n%s" % (cls.__name__, rdf, " " * indent))
            self._pprint_dict(object, stream, indent, allowance + 1, context, level)
            stream.write(")")

        _dispatch[_collections.defaultdict.__repr__] = _pprint_default_dict

        def _pprint_counter(self, object, stream, indent, allowance, context, level):
            if not len(object):
                stream.write(repr(object))
                return
            cls = object.__class__
            stream.write(cls.__name__ + "({")
            if self._indent_per_level > 1:
                stream.write((self._indent_per_level - 1) * " ")
            items = object.most_common()
            self._format_dict_items(items, stream, indent + len(cls.__name__) + 1, allowance + 2, context, level)
            stream.write("})")

        _dispatch[_collections.Counter.__repr__] = _pprint_counter

        def _pprint_chain_map(self, object, stream, indent, allowance, context, level):
            if not len(object.maps):
                stream.write(repr(object))
                return
            cls = object.__class__
            stream.write(cls.__name__ + "(")
            indent += len(cls.__name__) + 1
            for i, m in enumerate(object.maps):
                if i == len(object.maps) - 1:
                    self._format(m, stream, indent, allowance + 1, context, level)
                    stream.write(")")
                else:
                    self._format(m, stream, indent, 1, context, level)
                    stream.write(",\n" + " " * indent)

        _dispatch[_collections.ChainMap.__repr__] = _pprint_chain_map

        def _pprint_deque(self, object, stream, indent, allowance, context, level):
            if not len(object):
                stream.write(repr(object))
                return
            cls = object.__class__
            stream.write(cls.__name__ + "(")
            indent += len(cls.__name__) + 1
            stream.write("[")
            if object.maxlen is None:
                self._format_items(object, stream, indent, allowance + 2, context, level)
                stream.write("])")
            else:
                self._format_items(object, stream, indent, 2, context, level)
                rml = self._repr(object.maxlen, context, level)
                stream.write("],\n%smaxlen=%s)" % (" " * indent, rml))

        _dispatch[_collections.deque.__repr__] = _pprint_deque

        def _pprint_user_dict(self, object, stream, indent, allowance, context, level):
            self._format(object.data, stream, indent, allowance, context, level - 1)

        _dispatch[_collections.UserDict.__repr__] = _pprint_user_dict

        def _pprint_user_list(self, object, stream, indent, allowance, context, level):
            self._format(object.data, stream, indent, allowance, context, level - 1)

        _dispatch[_collections.UserList.__repr__] = _pprint_user_list

        def _pprint_user_string(self, object, stream, indent, allowance, context, level):
            self._format(object.data, stream, indent, allowance, context, level - 1)

        _dispatch[_collections.UserString.__repr__] = _pprint_user_string


# Return triple (repr_string, isreadable, isrecursive).


def _safe_repr(object, context, maxlevels, level, sort_dicts):
    typ = type(object)
    if typ in _builtin_scalars:
        return repr(object), True, False

    r = getattr(typ, "__repr__", None)
    if issubclass(typ, dict) and r is dict.__repr__:
        if not object:
            return "{}", True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return _recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        if sort_dicts:
            items = sorted(object.items(), key=_safe_tuple)
        else:
            items = object.items()
        for k, v in items:
            krepr, kreadable, krecur = _safe_repr(k, context, maxlevels, level, sort_dicts)
            vrepr, vreadable, vrecur = _safe_repr(v, context, maxlevels, level, sort_dicts)
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or (issubclass(typ, tuple) and r is tuple.__repr__):
        if issubclass(typ, list):
            if not object:
                return "[]", True, False
            format = "[%s]"
        elif len(object) == 1:
            format = "(%s,)"
        else:
            if not object:
                return "()", True, False
            format = "(%s)"
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return format % "...", False, objid in context
        if objid in context:
            return _recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        append = components.append
        level += 1
        for o in object:
            orepr, oreadable, orecur = _safe_repr(o, context, maxlevels, level, sort_dicts)
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    rep = repr(object)
    return rep, (rep and not rep.startswith("<")), False


_builtin_scalars = frozenset({str, bytes, bytearray, int, float, complex, bool, type(None)})


def _recursion(object):
    return "<Recursion on %s with id=%s>" % (type(object).__name__, id(object))


def _perfcheck(object=None):
    import time

    if object is None:
        object = [("string", (1, 2), [3, 4], {5: 6, 7: 8})] * 100000
    p = PrettyPrinter()
    t1 = time.perf_counter()
    _safe_repr(object, {}, None, 0, True)
    t2 = time.perf_counter()
    p.pformat(object)
    t3 = time.perf_counter()
    print("_safe_repr:", t2 - t1)
    print("pformat:", t3 - t2)


def _wrap_bytes_repr(object, width, allowance):
    current = b""
    last = len(object) // 4 * 4
    for i in range(0, len(object), 4):
        part = object[i : i + 4]
        candidate = current + part
        if i == last:
            width -= allowance
        if len(repr(candidate)) > width:
            if current:
                yield repr(current)
            current = part
        else:
            current = candidate
    if current:
        yield repr(current)


# end of source of pprint (3.8)

if PY2:
    import pprint

    pformat = pprint.pformat

