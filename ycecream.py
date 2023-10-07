from __future__ import print_function

#   _   _   ___   ___   ___  _ __   ___   __ _  _ __ ___
#  | | | | / __| / _ \ / __|| '__| / _ \ / _` || '_ ` _ \
#   \__, || (__ |  __/| (__ | |   |  __/| (_| || | | | | |
#   |___/  \___| \___| \___||_|    \___| \__,_||_| |_| |_|
#                       sweeter debugging and benchmarking

__version__ = "1.3.15"

"""
See https://github.com/salabim/ycecream for details

(c)2023 Ruud van der Ham - rt.van.der.ham@gmail.com

Inspired by IceCream "Never use print() to debug again".
Also contains some of the original code.
IceCream was written by Ansgar Grunseid / grunseid.com / grunseid@gmail.com
"""


def copy_contents(package, prefer_installed, filecontents):
    import tempfile
    import shutil
    import sys
    from pathlib import Path
    import zlib
    import base64
    if package in sys.modules:
        return
    if prefer_installed:
        for dir in sys.path:
            dir = Path(dir)
            if (dir / package).is_dir() and (dir / package / '__init__.py').is_file():
                return
            if (dir / (package + '.py')).is_file():
                return
    target_dir = Path(tempfile.gettempdir()) / ('embedded_' + package) 
    if target_dir.is_dir():
        shutil.rmtree(target_dir, ignore_errors=True)
    for file, contents in filecontents:
        ((target_dir / file).parent).mkdir(parents=True, exist_ok=True)
        with open(target_dir / file, 'wb') as f:
            f.write(zlib.decompress(base64.b64decode(contents)))
    sys.path.insert(prefer_installed * len(sys.path), str(target_dir))
del copy_contents

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
import executing

nv = object()

PY2 = sys.version_info.major == 2
PY3 = sys.version_info.major == 3

if PY2:

    def ycecream_pformat(obj, width, indent, depth):
        return pformat(obj, width=width, indent=indent, depth=depth).replace(
            "\\n", "\n"
        )


if PY3:

    def perf_counter():
        return (
            time.perf_counter() if _fixed_perf_counter is None else _fixed_perf_counter
        )

    from pathlib import Path

    def ycecream_pformat(obj, width, compact, indent, depth, sort_dicts):
        return pformat(
            obj,
            width=width,
            compact=compact,
            indent=indent,
            depth=depth,
            sort_dicts=sort_dicts,
        ).replace("\\n", "\n")


class Source(executing.Source):
    def get_text_with_indentation(self, node):
        result = self.asttokens().get_text(node)
        if "\n" in result:
            result = " " * node.first_token.start[1] + result
            result = dedent(result)
        result = result.strip()
        return result


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
    "o": "output",
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
    "dl": "delta",
}


def set_defaults():
    default.prefix = "y| "
    default.output = "stderr"
    default.serialize = (
        ycecream_pformat  # can't use pformat directly as that is defined later
    )
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
            raise ValueError(
                "error in {json_file}: key {k} not allowed".format(
                    json_file=json_file, k=k
                )
            )

        if k in shortcut_to_name:
            k = shortcut_to_name[k]
        if hasattr(default, k):
            setattr(default, k, v)
        else:
            if k == "delta":
                setattr(default, "start_time", perf_counter() - v)
            else:
                raise ValueError(
                    "error in {json_file}: key {k} not recognized".format(
                        json_file=json_file, k=k
                    )
                )


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
        #     decorator=nv,
        #     context_manager=nv,
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
                        raise ValueError(
                            "can't use {key} and {full_name} in {func}".format(
                                key=key, full_name=full_name, func=func
                            )
                        )
            else:
                raise TypeError(
                    "{func} got an unexpected keyword argument {key}".format(
                        func=func, key=key
                    )
                )
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
        d = kwargs.pop("decorator", nv)
        context_manager = kwargs.pop("context_manager", nv)
        cm = kwargs.pop("cm", nv)
        delta = kwargs.pop("delta", nv)
        as_str = kwargs.pop("as_str", nv)
        provided = kwargs.pop("provided", nv)
        pr = kwargs.pop("pr", nv)

        if d is not nv and decorator is not nv:
            raise TypeError("can't use both d and decorator")
        if cm is not nv and context_manager is not nv:
            raise TypeError("can't use both cm and context_manager")
        if pr is not nv and provided is not nv:
            raise TypeError("can't use both pr and provided")

        as_str = False if as_str is nv else bool(as_str)
        provided = True if provided is nv else bool(provided)
        decorator = False if decorator is nv else bool(decorator)
        context_manager = False if context_manager is nv else bool(context_manager)

        if decorator and context_manager:
            raise TypeError("decorator and context_manager can't be specified both.")

        self.is_context_manager = False

        Pair = collections.namedtuple("Pair", "left right")

        this = self.fork()
        this.assign(kwargs, locals(), func="__call__")

        if this.enabled == [] and not (
            as_str or this.decorator or this.context_manager
        ):
            return return_args(args, this.return_none)

        if not provided:
            this.enabled = False

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
            if (
                (filename.startswith("<") and filename.endswith(">"))
                or (main_file_resolved is None)
                or (filename_resolved == main_file_resolved)
            ):
                filename_name = ""
            else:
                filename_name = "[" + os.path.basename(filename) + "]"

            if filename not in codes:
                frame_info = inspect.getframeinfo(
                    call_frame, context=1000000
                )  # get the full source code
                if frame_info.code_context is None:
                    no_source_error()
                codes[filename] = frame_info.code_context
            code = codes[filename]
            frame_info = inspect.getframeinfo(call_frame, context=1)

            parent_function = frame_info.function  # changed in version 1.3.10 ****
            parent_function = Source.executing(call_frame).code_qualname()
            parent_function = parent_function.replace(".<locals>.", ".")
            if parent_function == "<module>" or str(this.show_line_number) in (
                "n",
                "no parent",
            ):
                parent_function = ""
            else:
                parent_function = " in {parent_function}()".format(
                    parent_function=parent_function
                )
            line_number = frame_info.lineno
            if 0 <= line_number - 1 < len(code):
                this_line = code[line_number - 1].strip()
            else:
                this_line = ""
            if 0 <= line_number - 2 < len(code):
                this_line_prev = code[line_number - 2].strip()
            else:
                this_line_prev = ""
        if (
            this_line.startswith("@")
            or this_line_prev.startswith("@")
            or this.decorator
        ):
            if as_str:
                raise TypeError("as_str may not be True when y used as decorator")

            for ln, line in enumerate(code[line_number - 1 :], line_number):
                if line.strip().startswith("def") or line.strip().startswith("class"):
                    line_number = ln
                    break
            else:
                line_number += 1
            this.line_number_with_filename_and_parent = (
                "#{line_number}{filename_name}{parent_function}".format(
                    line_number=line_number,
                    filename_name=filename_name,
                    parent_function=parent_function,
                )
            )

            def real_decorator(function):
                @functools.wraps(function)
                def wrapper(*args, **kwargs):
                    enter_time = perf_counter()
                    context = this.context()

                    args_kwargs = [repr(arg) for arg in args] + [
                        "{k}={repr_v}".format(k=k, repr_v=repr(v))
                        for k, v in kwargs.items()
                    ]
                    function_arguments = (
                        function.__name__ + "(" + (", ".join(args_kwargs)) + ")"
                    )

                    if this.show_enter:
                        this.do_output(
                            "{context}called {function_arguments}{traceback}".format(
                                context=context,
                                function_arguments=function_arguments,
                                traceback=this.traceback(),
                            )
                        )
                    result = function(*args, **kwargs)
                    duration = perf_counter() - enter_time

                    context = this.context()
                    if this.show_exit:
                        this.do_output(
                            "{context}returned {repr_result} from {function_arguments} in {duration:.6f} seconds{traceback}".format(
                                context=context,
                                repr_result=repr(result),
                                function_arguments=function_arguments,
                                duration=duration,
                                traceback=this.traceback(),
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

            this.line_number_with_filename_and_parent = (
                "#{line_number}{filename_name}{parent_function}".format(
                    line_number=line_number,
                    filename_name=filename_name,
                    parent_function=parent_function,
                )
            )

        if (
            this_line.startswith("with ")
            or this_line.startswith("with\t")
            or this.context_manager
        ):
            if as_str:
                raise TypeError("as_str may not be True when y used as context manager")
            if args:
                raise TypeError(
                    "non-keyword arguments are not allowed when y used as context manager"
                )

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
                                s = ast.parse(left, mode="eval")
                            if isinstance(s, ast.Expression):
                                s = s.body
                            if s and isinstance(
                                s, ast.JoinedStr
                            ):  # it is indeed an f-string
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
                    as_one_line = context + this.separator.join(
                        pair.left + this.serialize_kwargs(obj=pair.right, width=10000)
                        for pair in pairs
                    )
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
                        line = start + this.serialize_kwargs(
                            obj=pair.right, width=this.line_length - len(start)
                        )
                        if "\n" in line:
                            lines.append(start)
                            do_right = True
                        else:
                            lines.append(line)
                    indent1 = indent1_rest
                    if do_right:
                        indent2 = indent1 + wrap_indent
                        line = this.serialize_kwargs(
                            obj=pair.right, width=this.line_length - len(indent2)
                        )
                        for s in line.splitlines():
                            lines.append(indent2 + s)

                out = "\n".join(line.rstrip() for line in lines)

        else:
            if not this.show_line_number:  # if "n" or "no parent", keep that info
                this.show_line_number = True
            out = this.context(omit_context_separator=True)

        if this.show_traceback:
            out += this.traceback()

        if as_str:
            if this.enabled:
                if this.enforce_line_length:
                    out = "\n".join(
                        line[: this.line_length] for line in out.splitlines()
                    )
                return out + "\n"
            else:
                return ""
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
        #        decorator=nv,
        #        context_manager=nv,
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
        #        decorator=nv,
        #        context_manager=nv,
        delta=nv,
        **kwargs
    ):
        this = _Y(_parent=self._parent)
        this.assign({}, self._attributes, func="clone()")
        this.assign(kwargs, locals(), func="clone()")

        return this

    def assert_(self, condition):
        if self.enabled:
            assert condition

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
            self.do_output(
                "{context}exit in {duration:.6f} seconds{traceback}".format(
                    context=context, duration=duration, traceback=self.save_traceback
                )
            )
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
                wrap_indent
                + '  File "'
                + entry[0]
                + '", line '
                + str(entry[1])
                + ", in "
                + entry[2]
                + "\n"
                + wrap_indent
                + "    "
                + entry[3]
                for entry in traceback.extract_stack()[:-2]
            )
        else:
            return ""

    def check(self):

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

            kwargs = {
                key: getattr(self, key)
                for key in ("indent", "depth")
                if key in inspect.getargspec(self.serialize).args
            }
            kwargs["width"] = width

            return self.serialize(obj, **kwargs)

    if PY3:

        def serialize_kwargs(self, obj, width):
            kwargs = {
                key: getattr(self, key)
                for key in ("sort_dicts", "compact", "indent", "depth")
                if key in inspect.signature(self.serialize).parameters
            }
            if "width" in inspect.signature(self.serialize).parameters:
                kwargs["width"] = width
            return self.serialize(obj, **kwargs)


codes = {}

set_defaults()
default_pre_json = copy.copy(default)
apply_json()
y = _Y()
yc = y.fork(prefix="yc| ")


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


def pprint(
    object, stream=None, indent=1, width=80, depth=None, compact=False, sort_dicts=True
):
    """Pretty-print a Python object to a stream [default is sys.stdout]."""
    printer = PrettyPrinter(
        stream=stream,
        indent=indent,
        width=width,
        depth=depth,
        compact=compact,
        sort_dicts=sort_dicts,
    )
    printer.pprint(object)


def pformat(object, indent=1, width=80, depth=None, compact=False, sort_dicts=True):
    """Format a Python object into a pretty-printed representation."""
    return PrettyPrinter(
        indent=indent, width=width, depth=depth, compact=compact, sort_dicts=sort_dicts
    ).pformat(object)


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
            return (str(type(self.obj)), id(self.obj)) < (
                str(type(other.obj)),
                id(other.obj),
            )


def _safe_tuple(t):
    "Helper function for comparing 2-tuples"
    return _safe_key(t[0]), _safe_key(t[1])


if PY3:

    class PrettyPrinter:
        def __init__(
            self,
            indent=1,
            width=80,
            depth=None,
            stream=None,
            compact=False,
            sort_dicts=True,
        ):
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
                    self._pprint_dict(
                        object, stream, indent, allowance, context, level + 1
                    )
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
                self._format_dict_items(
                    items, stream, indent, allowance + 1, context, level
                )
            write("}")

        _dispatch[dict.__repr__] = _pprint_dict

        def _pprint_ordered_dict(
            self, object, stream, indent, allowance, context, level
        ):
            if not len(object):
                stream.write(repr(object))
                return
            cls = object.__class__
            stream.write(cls.__name__ + "(")
            self._format(
                list(object.items()),
                stream,
                indent + len(cls.__name__) + 1,
                allowance + 1,
                context,
                level,
            )
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
            self._format_items(
                object, stream, indent, allowance + len(endchar), context, level
            )
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
            self._format_items(
                object, stream, indent, allowance + len(endchar), context, level
            )
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
            self._pprint_bytes(
                bytes(object), stream, indent + 10, allowance + 1, context, level + 1
            )
            write(")")

        _dispatch[bytearray.__repr__] = _pprint_bytearray

        def _pprint_mappingproxy(
            self, object, stream, indent, allowance, context, level
        ):
            stream.write("mappingproxy(")
            self._format(
                object.copy(), stream, indent + 13, allowance + 1, context, level
            )
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
                self._format(
                    ent,
                    stream,
                    indent + len(rep) + 2,
                    allowance if last else 1,
                    context,
                    level,
                )
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
                self._format(
                    ent, stream, indent, allowance if last else 1, context, level
                )

        def _repr(self, object, context, level):
            repr, readable, recursive = self.format(
                object, context.copy(), self._depth, level
            )
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

        def _pprint_default_dict(
            self, object, stream, indent, allowance, context, level
        ):
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
            self._format_dict_items(
                items,
                stream,
                indent + len(cls.__name__) + 1,
                allowance + 2,
                context,
                level,
            )
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
                self._format_items(
                    object, stream, indent, allowance + 2, context, level
                )
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

        def _pprint_user_string(
            self, object, stream, indent, allowance, context, level
        ):
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
            krepr, kreadable, krecur = _safe_repr(
                k, context, maxlevels, level, sort_dicts
            )
            vrepr, vreadable, vrecur = _safe_repr(
                v, context, maxlevels, level, sort_dicts
            )
            append("%s: %s" % (krepr, vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return "{%s}" % ", ".join(components), readable, recursive

    if (issubclass(typ, list) and r is list.__repr__) or (
        issubclass(typ, tuple) and r is tuple.__repr__
    ):
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
            orepr, oreadable, orecur = _safe_repr(
                o, context, maxlevels, level, sort_dicts
            )
            append(orepr)
            if not oreadable:
                readable = False
            if orecur:
                recursive = True
        del context[objid]
        return format % ", ".join(components), readable, recursive

    rep = repr(object)
    return rep, (rep and not rep.startswith("<")), False


_builtin_scalars = frozenset(
    {str, bytes, bytearray, int, float, complex, bool, type(None)}
)


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

