#   _   _   ___   ___   ___  _ __   ___   __ _  _ __ ___
#  | | | | / __| / _ \ / __|| '__| / _ \ / _` || '_ ` _ \
#  | |_| || (__ |  __/| (__ | |   |  __/| (_| || | | | | |
#   \__, | \___| \___| \___||_|    \___| \__,_||_| |_| |_|
#   |___/     the no depencency, Pythonic fork of IceCream
#
#  See https://raw.githubusercontent.com/salabim/ycecream/master/readme.md for details

__version__ = "1.0.2"

"""
Fork from IceCream - Never use print() to debug again
Original author: Ansgar Grunseid / grunseid.com / grunseid@gmail.com

(c)2021 Ruud van der Ham - rt.van.der.ham@gmail.com
"""

import ast
import inspect
import pprint
import sys
import datetime
import textwrap
from pathlib import Path


def stderr_print(*args):
    print(*args, file=sys.stderr)


def isLiteral(s):
    try:
        ast.literal_eval(s)
    except Exception:
        return False
    return True


class NoSourceAvailableError(OSError):
    infoMessage = (
        "Failed to access the underlying source code for analysis. Was y() "
        "invoked in an interpreter (e.g. python -i), a frozen application "
        "(e.g. packaged with PyInstaller), or did the underlying source code "
        "change during execution?"
    )


def callOrValue(obj):
    return obj() if callable(obj) else obj


def prefixLinesAfterFirst(prefix, s):
    lines = s.splitlines(True)

    for i in range(1, len(lines)):
        lines[i] = prefix + lines[i]

    return "".join(lines)


def indented_lines(prefix, string):
    lines = string.splitlines()
    return [prefix + lines[0]] + [" " * len(prefix) + line for line in lines[1:]]


def format_pair(prefix, arg, value):
    arg_lines = indented_lines(prefix, arg)
    value_prefix = arg_lines[-1] + ": "

    looksLikeAString = value[0] + value[-1] in ["''", '""']
    if looksLikeAString:  # Align the start of multiline strings.
        value = prefixLinesAfterFirst(" ", value)

    value_lines = indented_lines(value_prefix, value)
    lines = arg_lines[:-1] + value_lines
    return "\n".join(lines)


PREFIX = "y| "
LINE_WRAP_WIDTH = 80  # Characters.
CONTEXT_DELIMITER = " ==> "
OUTPUT_FUNCTION = lambda *args: stderr_print(*args)
ARG_TO_STRING_FUNCTION = lambda obj: pprint.pformat(obj).replace("\\n", "\n")
INCLUDE_CONTEXT = False
INCLUDE_TIME = False
INCLUDE_DELTA = False

ENABLED = True
PAIR_DELIMITER = ", "

starttime = datetime.datetime.now()


class Y:

    def __init__(
        self,
        prefix=PREFIX,
        output_function=OUTPUT_FUNCTION,
        arg_to_string_function=ARG_TO_STRING_FUNCTION,
        include_context=INCLUDE_CONTEXT,
        include_time=INCLUDE_TIME,
        include_delta=INCLUDE_DELTA,
        line_wrap_width=LINE_WRAP_WIDTH,
        pair_delimiter=PAIR_DELIMITER,
        enabled=ENABLED,
    ):

        self.prefix = PREFIX if prefix is None else prefix
        self.output_function = OUTPUT_FUNCTION if output_function is None else output_function
        self.arg_to_string_function = ARG_TO_STRING_FUNCTION if arg_to_string_function is None else arg_to_string_function
        self.include_context = INCLUDE_CONTEXT if include_context is None else include_context
        self.include_time = INCLUDE_TIME if include_time is None else include_time
        self.include_delta = INCLUDE_DELTA if include_delta is None else include_delta
        self.line_wrap_width = LINE_WRAP_WIDTH if line_wrap_width is None else line_wrap_width
        self.pair_delimiter = PAIR_DELIMITER if pair_delimiter is None else pair_delimiter
        self.enabled = ENABLED if enabled is None else enabled

    def __call__(self, *args):
        if self.enabled:
            call_frame = inspect.currentframe().f_back
            try:
                out = self._format(call_frame, *args)
            except NoSourceAvailableError as err:
                prefix = callOrValue(self.prefix)
                out = prefix + "Error: " + err.infoMessage
            self.output_function(out)

        if not args:  # E.g. ic().
            passthrough = None
        elif len(args) == 1:  # E.g. ic(1).
            passthrough = args[0]
        else:  # E.g. ic(1, 2, 3).
            passthrough = args

        return passthrough

    def as_str(self, *args):
        call_frame = inspect.currentframe().f_back
        out = self._format(call_frame, *args)
        return out

    def _format(self, call_frame, *args):
        prefix = callOrValue(self.prefix)

        call_node = Source.executing(call_frame).node
        if call_node is None:
            raise NoSourceAvailableError()

        if len(args) == 0 or self.include_context:
            parts = [self._format_context(call_frame, call_node)]
        else:
            parts = []
        if self.include_time:
            parts.append(f'@ {datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}')

        if self.include_delta:
            t0 = (datetime.datetime.now() - starttime).total_seconds()
            parts.append(f"\u0394 {t0:.3f}")

        context = " ".join(parts)

        if args:
            return self._format_args(call_frame, call_node, prefix, context, args)
        else:
            return prefix + context

    def _format_args(self, call_frame, call_node, prefix, context, args):
        source = Source.for_frame(call_frame)
        sanitized_args = [source.get_text_with_indentation(arg) for arg in call_node.args]

        pairs = list(zip(sanitized_args, args))

        out = self._construct_argument_output(prefix, context, pairs)
        return out

    def _construct_argument_output(self, prefix, context, pairs):
        def arg_prefix(arg):
            return f"{arg}: "

        pairs = [(arg, self.arg_to_string_function(val)) for arg, val in pairs]
        pairs_processed = [val if isLiteral(arg) else (arg_prefix(arg) + val) for arg, val in pairs]

        all_args_on_one_line = self.pair_delimiter.join(pairs_processed)
        multiline_args = len(all_args_on_one_line.splitlines()) > 1

        context_delimiter = CONTEXT_DELIMITER if context else ""
        all_pairs = prefix + context + context_delimiter + all_args_on_one_line
        first_line_too_long = len(all_pairs.splitlines()[0]) > self.line_wrap_width

        if multiline_args or first_line_too_long:
            if context:
                lines = [prefix + context] + [format_pair(len(prefix) * " ", arg, value) for arg, value in pairs]
            else:
                arg_lines = [format_pair("", arg, value) for arg, value in pairs]
                lines = indented_lines(prefix, "\n".join(arg_lines))
        else:
            lines = [prefix + context + context_delimiter + all_args_on_one_line]

        return "\n".join(lines)

    def _format_context(self, call_frame, call_node):
        filename, line_number, parent_function = self._get_context(call_frame, call_node)

        if parent_function != "<module>":
            parent_function = f"{parent_function}()"

        context = f"{filename}:{line_number} in {parent_function}"
        return context

    def _get_context(self, call_frame, call_node):
        line_number = call_node.lineno
        frame_info = inspect.getframeinfo(call_frame)
        parent_function = frame_info.function

        filename = Path(frame_info.filename).name

        return filename, line_number, parent_function

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def given(
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
        return  Y(
        prefix=self.prefix if prefix is None else prefix,
        output_function = self.output_function if output_function is None else output_function,
        arg_to_string_function = self.arg_to_string_function if arg_to_string_function is None else arg_to_string_function,
        include_context = self.include_context if include_context is None else include_context,
        include_time = self.include_time if include_time is None else include_time,
        include_delta = self.include_delta if include_delta is None else include_delta,
        line_wrap_width = self.line_wrap_width if line_wrap_width is None else line_wrap_width,
        pair_delimiter = self.pair_delimiter if pair_delimiter is None else pair_delimiter,
        enabled = self.enabled if enabled is None else enabled
        )

try:
    builtins = __import__("__builtin__")
except ImportError:
    builtins = __import__("builtins")


def install(y="y"):
    setattr(builtins, y, getattr("ycecream", y))


def uninstall(y="y"):
    delattr(builtins, y)


ic = Y(prefix="ic| ")
y = Y(prefix="y| ")

# source of asttokens.util

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

# from six import iteritems


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
    c
    for n, c in ast.__dict__.items()
    if isinstance(c, type) and issubclass(c, (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop))  # iteritems changed into items
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


stmt_class_names = {n for n, c in ast.__dict__.items() if isinstance(c, type) and issubclass(c, ast.stmt)}  # changed iteritems into items
expr_class_names = {n for n, c in ast.__dict__.items() if isinstance(c, type) and issubclass(c, ast.expr)} | {  # changed iteritems into items
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


# end of source of asttokens.util


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

# import six


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

    #  if six.PY2:
    #    # We shouldn't do this on PY3 because its SetComp/DictComp already have a correct start.
    #    def visit_setcomp(self, node, first_token, last_token):
    #      return self.handle_comp('{', node, first_token, last_token)
    #
    #    def visit_dictcomp(self, node, first_token, last_token):
    #      return self.handle_comp('{', node, first_token, last_token)

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
        elif isinstance(node.value, (str, bytes)):
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

    #  if six.PY2:
    #    # No need for this on Python3, which already handles 'with' nodes correctly.
    #    def visit_with(self, node, first_token, last_token):
    #      first = self._code.find_token(first_token, token.NAME, 'with', reverse=True)
    #      return (first, last_token)

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


# end of source of line numbers


# source of asttokens.astokens
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

# import six
# from six.moves import xrange      # pylint: disable=redefined-builtin


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
        if isinstance(source_text, bytes):
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
        for i in range(first_token.index, last_token.index + 1):  # changed xrange in range
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


# end of source of asttokes

# source of executing module
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

from functools import lru_cache
from tokenize import detect_encoding
from itertools import zip_longest
from pathlib import Path

cache = lru_cache(maxsize=None)
text_type = str

try:
    _get_instructions = dis.get_instructions
except AttributeError:

    class Instruction(namedtuple("Instruction", "offset argval opname starts_line")):
        lineno = None

    from dis import HAVE_ARGUMENT, EXTENDED_ARG, hasconst, opname, findlinestarts

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
    if not condition:
        raise AssertionError(str(message))


def get_instructions(co):
    lineno = None
    for inst in _get_instructions(co):
        lineno = inst.starts_line or lineno
        assert_(lineno)
        inst.lineno = lineno
        yield inst


class NotOneValueFound(Exception):
    pass


def only(it):
    if hasattr(it, "__len__"):
        if len(it) != 1:
            raise NotOneValueFound(f"Expected one value, found {len(it)}")
        return list(it)[0]

    lst = tuple(islice(it, 2))
    if len(lst) == 0:
        raise NotOneValueFound("Expected one value, found 0")
    if len(lst) > 1:
        raise NotOneValueFound("Expected one value, found several")
    return lst[0]


class Source(object):
    def __init__(self, filename, lines):

        self.filename = filename
        text = "".join(lines)

        if not isinstance(text, text_type):
            encoding = self.detect_encoding(text)
            text = text.decode(encoding)
            lines = [line.decode(encoding) for line in lines]

        self.text = text
        self.lines = [line.rstrip("\r\n") for line in lines]

        ast_text = text
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
        if isinstance(frame_or_tb, types.TracebackType):
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
                        if retry_cache and isinstance(e, (NotOneValueFound, AssertionError)):
                            return find(source=cls.for_frame(frame, use_cache=False), retry_cache=False)

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
        result = cls.__dict__.get(name, default)
        setattr(cls, name, result)
        return result

    @cache
    def statements_at_line(self, lineno):
        return {statement_containing_node(node) for node in self._nodes_by_line[lineno]}

    @cache
    def asttokens(self):
        #        from asttokens import ASTTokens

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
        assert_(code.co_filename == self.filename)
        return self._qualnames.get((code.co_name, code.co_firstlineno), code.co_name)


class Executing(object):
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
        if not any(inst.opname == "JUMP_IF_NOT_DEBUG" for inst in self.compile_instructions()):
            result = [inst for inst in result if inst.opname != "JUMP_IF_NOT_DEBUG"]

        return result

    def matching_nodes(self, exprs):
        original_instructions = self.get_original_clean_instructions()
        original_index = only(i for i, inst in enumerate(original_instructions) if inst.offset == self.lasti)
        for i, expr in enumerate(exprs):
            setter = get_setter(expr)
            replacement = ast.BinOp(left=expr, op=ast.Pow(), right=ast.Str(s=sentinel))
            ast.fix_missing_locations(replacement)
            setter(replacement)
            try:
                instructions = self.compile_instructions()
            finally:
                setter(expr)
            indices = [i for i, instruction in enumerate(instructions) if instruction.argval == sentinel]

            for index_num, sentinel_index in enumerate(indices):
                sentinel_index -= index_num * 2

                assert_(instructions.pop(sentinel_index).opname == "LOAD_CONST")
                assert_(instructions.pop(sentinel_index).opname == "BINARY_POWER")

            for index_num, sentinel_index in enumerate(indices):
                sentinel_index -= index_num * 2
                new_index = sentinel_index - 1

                if new_index != original_index:
                    continue

                original_inst = original_instructions[original_index]
                new_inst = instructions[new_index]
                if (
                    original_inst.opname == new_inst.opname in ("CONTAINS_OP", "IS_OP")
                    and original_inst.arg != new_inst.arg
                    and (original_instructions[original_index + 1].opname != instructions[new_index + 1].opname == "UNARY_NOT")
                ):
                    instructions.pop(new_index + 1)

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
    stmt = list(stmts)[0]
    while not isinstance(stmt.parent, ast.Module):
        stmt = stmt.parent
    tree = ast.parse("")
    tree.body = [stmt]
    ast.copy_location(tree, stmt)
    return tree


# end of source of executing module

class Source(Source): 
    def get_text_with_indentation(self, node):
        result = self.asttokens().get_text(node)
        if "\n" in result:
            result = " " * node.first_token.start[1] + result
            result = textwrap.dedent(result)
        result = result.strip()
        return result
        
