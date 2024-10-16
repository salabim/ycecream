from __future__ import print_function
from __future__ import division

#  In order to tun under Python 2.7, the following packages have to be pip-installed:
#      pathlib
#      backports.tempfile

import sys
from pathlib import Path
from ycecream import y
import ycecream
import datetime
import time
import pytest

PY2 = sys.version_info.major == 2
PY3 = sys.version_info.major == 3


if PY2:
    from backports import tempfile

if PY3:
    import tempfile


class g:
    pass


context_start = "y| #"


y = y.new(ignore_json=True)


if PY2:
    ycecream.change_path(Path)


FAKE_TIME = datetime.datetime(2021, 1, 1, 0, 0, 0)


@pytest.fixture
def patch_datetime_now(monkeypatch):
    class mydatetime:
        @classmethod
        def now(cls):
            return FAKE_TIME

    monkeypatch.setattr(datetime, "datetime", mydatetime)


def test_time(patch_datetime_now):
    hello = "world"
    s = y(hello, show_time=True, as_str=True)
    assert s == "y| @ 00:00:00.000000 ==> hello: 'world'\n"


def test_no_arguments(capsys):
    result = y()
    out, err = capsys.readouterr()
    assert err.startswith(context_start)
    assert err.endswith(" in test_no_arguments()\n")
    assert result is None


def test_one_arguments(capsys):
    hello = "world"
    result = y(hello)
    y(hello)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| hello: 'world'
y| hello: 'world'
"""
    )
    assert result == hello


def test_two_arguments(capsys):
    hello = "world"
    ll = [1, 2, 3]
    result = y(hello, ll)
    out, err = capsys.readouterr()
    assert err == "y| hello: 'world', ll: [1, 2, 3]\n"
    assert result == (hello, ll)


def test_in_function(capsys):
    def hello(val):
        y(val, show_line_number=True)

    hello("world")
    out, err = capsys.readouterr()
    assert err.startswith(context_start)
    assert err.endswith(" in test_in_function.hello() ==> val: 'world'\n")


def test_in_function_no_parent(capsys):
    def hello(val):
        y(val, show_line_number="n")

    hello("world")
    out, err = capsys.readouterr()
    assert err.startswith(context_start)
    assert not err.endswith(" in test_in_function_no_parent.hello() ==> val: 'world'\n")


def test_prefix(capsys):
    hello = "world"
    y(hello, prefix="==> ")
    out, err = capsys.readouterr()
    assert err == "==> hello: 'world'\n"


def test_time_delta():
    sdelta0 = y(1, show_delta=True, as_str=True)
    stime0 = y(1, show_time=True, as_str=True)
    time.sleep(0.001)
    sdelta1 = y(1, show_delta=True, as_str=True)
    stime1 = y(1, show_time=True, as_str=True)
    assert sdelta0 != sdelta1
    assert stime0 != stime1
    y.delta = 10
    time.sleep(0.1)
    assert 10.05 < y.delta < 11


def test_dynamic_prefix(capsys):
    g.i = 0

    def prefix():
        g.i += 1
        return str(g.i) + ")"

    hello = "world"
    y(hello, prefix=prefix)
    y(hello, prefix=prefix)
    out, err = capsys.readouterr()
    assert err == "1)hello: 'world'\n2)hello: 'world'\n"


def test_values_only():
    with y.preserve():
        y.configure(values_only=True)
        hello = "world"
        s = y(hello, as_str=True)
        assert s == "y| 'world'\n"


def test_calls():
    with pytest.raises(TypeError):
        y.new(a=1)
    with pytest.raises(TypeError):
        y.clone(a=1)
    with pytest.raises(TypeError):
        y.configure(a=1)
    with pytest.raises(TypeError):
        y(12, a=1)
    with pytest.raises(TypeError):
        y(a=1)


def test_output(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:  # we can't use tmpdir from pytest because of Python 2.7 compatibity
        g.result = ""

        def my_output(s):
            g.result += s + "\n"

        hello = "world"
        y(hello, output=print)
        out, err = capsys.readouterr()
        assert out == "y| hello: 'world'\n"
        assert err == ""
        y(hello, output=sys.stdout)
        out, err = capsys.readouterr()
        assert out == "y| hello: 'world'\n"
        assert err == ""
        y(hello, output="stdout")
        out, err = capsys.readouterr()
        assert out == "y| hello: 'world'\n"
        assert err == ""
        y(hello, output="")
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""
        y(hello, output="null")
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""
        y(hello, output=print)
        out, err = capsys.readouterr()
        assert out == "y| hello: 'world'\n"
        assert err == ""

        if True:
            path = Path(tmpdir) / "x0"
            y(hello, output=path)
            out, err = capsys.readouterr()
            assert out == ""
            assert err == ""
            with path.open("r") as f:
                assert f.read() == "y| hello: 'world'\n"

            path = Path(tmpdir) / "x1"
            y(hello, output=path)
            out, err = capsys.readouterr()
            assert out == ""
            assert err == ""
            with path.open("r") as f:
                assert f.read() == "y| hello: 'world'\n"

            path = Path(tmpdir) / "x2"
            with path.open("a+") as f:
                y(hello, output=f)
            with pytest.raises(TypeError):  # closed file
                y(hello, output=f)
            out, err = capsys.readouterr()
            assert out == ""
            assert err == ""
            with path.open("r") as f:
                assert f.read() == "y| hello: 'world'\n"

        with pytest.raises(TypeError):
            y(hello, output=1)

        y(hello, output=my_output)
        y(1, output=my_output)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""
        assert g.result == "y| hello: 'world'\ny| 1\n"

    def test_serialize(capsys):
        def serialize(s):
            return repr(s) + " [len=" + str(len(s)) + "]"

        hello = "world"
        y(hello, serialize=serialize)
        out, err = capsys.readouterr()
        assert err == "y| hello: 'world' [len=5]\n"

    def test_show_time(capsys):
        hello = "world"
        y(hello, show_time=True)
        out, err = capsys.readouterr()
        assert err.endswith("hello: 'world'\n")
        assert "@ " in err

    def test_show_delta(capsys):
        hello = "world"
        y(hello, show_delta=True)
        out, err = capsys.readouterr()
        assert err.endswith("hello: 'world'\n")
        assert "delta=" in err

    def test_as_str(capsys):
        hello = "world"
        s = y(hello, as_str=True)
        y(hello)
        out, err = capsys.readouterr()
        assert err == s

        with pytest.raises(TypeError):

            @y(as_str=True)
            def add2(x):
                return x + 2

        with pytest.raises(TypeError):
            with y(as_str=True):
                pass


def test_clone():
    hello = "world"
    z = y.clone()
    z.configure(prefix="z| ")
    sy = y(hello, as_str=True)
    with y.preserve():
        y.configure(show_line_number=True)
        sz = z(hello, as_str=True)
        assert sy.replace("y", "z") == sz


def test_sort_dicts():
    world = {"EN": "world", "NL": "wereld", "FR": "monde", "DE": "Welt"}
    s0 = y(world, as_str=True)
    s1 = y(world, sort_dicts=False, as_str=True)
    s2 = y(world, sort_dicts=True, as_str=True)
    if PY2:
        assert s0 == s1 == s2 == "y| world: {'DE': 'Welt', 'EN': 'world', 'FR': 'monde', 'NL': 'wereld'}\n"
    if PY3:
        assert s0 == s1 == "y| world: {'EN': 'world', 'NL': 'wereld', 'FR': 'monde', 'DE': 'Welt'}\n"
        assert s2 == "y| world: {'DE': 'Welt', 'EN': 'world', 'FR': 'monde', 'NL': 'wereld'}\n"


def test_underscore_numbers():
    numbers = dict(x1=1, x2=1000, x3=1000000, x4=1234567890)    
    s0 = y(numbers, as_str=True)
    s1 = y(numbers, underscore_numbers=True, as_str=True)
    s2 = y(numbers, un=False, as_str=True)
    if PY2:
        assert s0 == s1 == s2 == "y| numbers: {'x1': 1, 'x2': 1000, 'x3': 1000000, 'x4': 1234567890}\n"
    if PY3:
        assert s0 == s2 == "y| numbers: {'x1': 1, 'x2': 1000, 'x3': 1000000, 'x4': 1234567890}\n"
        assert s1 == "y| numbers: {'x1': 1, 'x2': 1_000, 'x3': 1_000_000, 'x4': 1_234_567_890}\n"


def test_multiline():
    a = 1
    b = 2
    ll = list(range(15))
    # fmt: off
    s = y((a, b),
        [ll,
        ll], as_str=True)
    # fmt: on
    assert (
        s
        == """\
y|  (a, b): (1, 2)
    [ll,
    ll]:
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
"""
    )

    lines = "\n".join("line{i}".format(i=i) for i in range(4))
    result = y(lines, as_str=True)
    assert (
        result
        == """\
y|  lines:
        'line0
        line1
        line2
        line3'
"""
    )


def test_decorator(capsys):
    ycecream.fix_perf_counter(0)

    @y
    def mul(x, y):
        return x * y

    @y()
    def div(x, y):
        return x / y

    @y(show_enter=False)
    def add(x, y):
        return x + y

    @y(show_exit=False)
    def sub(x, y):
        return x - y

    @y(show_enter=False, show_exit=False)
    def pos(x, y):
        return x**y

    assert mul(2, 3) == 2 * 3
    assert div(10, 2) == 10 / 2
    assert add(2, 3) == 2 + 3
    assert sub(10, 2) == 10 - 2
    assert pow(10, 2) == 10**2
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| called mul(2, 3)
y| returned 6 from mul(2, 3) in 0.000000 seconds
y| called div(10, 2)
y| returned 5.0 from div(10, 2) in 0.000000 seconds
y| returned 5 from add(2, 3) in 0.000000 seconds
y| called sub(10, 2)
"""
    )
    ycecream.fix_perf_counter(None)


def test_decorator_edge_cases(capsys):
    ycecream.fix_perf_counter(0)

    @y
    def mul(x, y, factor=1):
        return x * y * factor

    assert mul(5, 6) == 30
    assert mul(5, 6, 10) == 300
    assert mul(5, 6, factor=10) == 300
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| called mul(5, 6)
y| returned 30 from mul(5, 6) in 0.000000 seconds
y| called mul(5, 6, 10)
y| returned 300 from mul(5, 6, 10) in 0.000000 seconds
y| called mul(5, 6, factor=10)
y| returned 300 from mul(5, 6, factor=10) in 0.000000 seconds
"""
    )
    ycecream.fix_perf_counter(None)


def test_decorator_with_methods(capsys):
    class Number:
        def __init__(self, value):
            self.value = value

        @y(show_exit=False)
        def __mul__(self, other):
            if isinstance(other, Number):
                return self.value * other.value
            else:
                return self.value * other

        def __repr__(self):
            return self.__class__.__name__ + "(" + str(self.value) + ")"

    a = Number(2)
    b = Number(3)
    print(a * 2)
    print(a * b)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| called __mul__(Number(2), 2)
y| called __mul__(Number(2), Number(3))
"""
    )
    assert (
        out
        == """4
6
"""
    )


def test_context_manager(capsys):
    ycecream.fix_perf_counter(0)
    with y():
        y(3)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| enter
y| 3
y| exit in 0.000000 seconds
"""
    )
    ycecream.fix_perf_counter(None)


def test_return_none(capsys):
    a = 2
    result = y(a, a)
    assert result == (a, a)
    result = y(a, a, return_none=True)
    assert result is None
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| a: 2, a: 2
y| a: 2, a: 2
"""
    )


def test_read_json1():
    with tempfile.TemporaryDirectory() as tmpdir:  # we can't use tmpdir from pytest because of Python 2.7 compatibity
        org_line_length = ycecream.default.line_length
        json_filename0 = Path(tmpdir) / "ycecream.json"
        with open(str(json_filename0), "w") as f:
            print('{"line_length":-1}', file=f)
        tmpdir1 = Path(tmpdir) / "ycecream"
        tmpdir1.mkdir()
        json_filename1 = Path(tmpdir1) / "ycecream.json"
        with open(str(json_filename1), "w") as f:
            print('{"line_length":-2}', file=f)
        save_sys_path = sys.path

        sys.path = [tmpdir] + [tmpdir1]
        ycecream.set_defaults()
        ycecream.apply_json()
        assert ycecream.default.line_length == -1

        sys.path = [str(tmpdir1)] + [tmpdir]
        ycecream.set_defaults()
        ycecream.apply_json()
        assert ycecream.default.line_length == -2

        sys.path = []
        ycecream.set_defaults()
        ycecream.apply_json()
        assert ycecream.default.line_length == 80

        with open(str(json_filename0), "w") as f:
            print('{"error":0}', file=f)

        sys.path = [tmpdir]
        with pytest.raises(ValueError):
            ycecream.set_defaults()
            ycecream.apply_json()

        sys.path = save_sys_path


def test_read_json2():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_filename = Path(tmpdir) / "ycecream.json"
        with open(str(json_filename), "w") as f:
            print('{"prefix": "xxx", "delta": 10}', file=f)

        sys.path = [tmpdir] + sys.path
        ycecream.set_defaults()
        ycecream.apply_json()
        sys.path.pop(0)

        y1 = y.new()

        s = y1(3, as_str=True)
        assert s == "xxx3\n"
        assert 10 < y1.delta < 11

        with open(str(json_filename), "w") as f:
            print('{"prefix1": "xxx"}', file=f)

        sys.path = [tmpdir] + sys.path
        with pytest.raises(ValueError):
            ycecream.set_defaults()
            ycecream.apply_json()
        sys.path.pop(0)

        with open(str(json_filename), "w") as f:
            print('{"serialize": "xxx"}', file=f)

        sys.path = [tmpdir] + sys.path
        with pytest.raises(ValueError):
            ycecream.set_defaults()
            ycecream.apply_json()

        sys.path.pop(0)

        tmpdir = Path(tmpdir) / "ycecream"
        tmpdir.mkdir()
        json_filename = Path(tmpdir) / "ycecream.json"
        with open(str(json_filename), "w") as f:
            print('{"prefix": "yyy"}', file=f)

        sys.path = [str(tmpdir)] + sys.path
        ycecream.set_defaults()
        ycecream.apply_json()
        sys.path.pop(0)

        y1 = y.new()

        s = y1(3, as_str=True)
        assert s == "yyy3\n"

        tmpdir = Path(tmpdir) / "ycecream"
        tmpdir.mkdir()
        json_filename = Path(tmpdir) / "ycecream.json"
        with open(str(json_filename), "w") as f:
            print("{}", file=f)

        sys.path = [str(tmpdir)] + sys.path
        ycecream.set_defaults()
        ycecream.apply_json()
        sys.path.pop(0)


def test_wrapping(capsys):
    if PY2:
        return

    l0 = "".join("         {c}".format(c=c) for c in "12345678") + "\n" + "".join(".........0" for c in "12345678")

    print(l0, file=sys.stderr)
    ccc = cccc = 3 * ["12345678123456789012"]
    ccc0 = [cccc[0] + "0"] + cccc[1:]
    y(ccc)
    y(cccc)
    y(ccc0)

    out, err = capsys.readouterr()
    assert (
        err
        == """\
         1         2         3         4         5         6         7         8
.........0.........0.........0.........0.........0.........0.........0.........0
y| ccc: ['12345678123456789012', '12345678123456789012', '12345678123456789012']
y|  cccc:
        ['12345678123456789012', '12345678123456789012', '12345678123456789012']
y|  ccc0:
        ['123456781234567890120',
         '12345678123456789012',
         '12345678123456789012']
"""
    )
    a = "1234"
    b = bb = 9 * ["123"]
    print(l0, file=sys.stderr)
    y(a, b)
    y(a, bb)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
         1         2         3         4         5         6         7         8
.........0.........0.........0.........0.........0.........0.........0.........0
y| a: '1234', b: ['123', '123', '123', '123', '123', '123', '123', '123', '123']
y|  a: '1234'
    bb: ['123', '123', '123', '123', '123', '123', '123', '123', '123']
"""
    )
    dddd = 10 * ["123"]
    dddd = ddddd = 10 * ["123"]
    e = "a\nb"
    print(l0, file=sys.stderr)
    y(a, dddd)
    y(a, ddddd)
    y(e)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
         1         2         3         4         5         6         7         8
.........0.........0.........0.........0.........0.........0.........0.........0
y|  a: '1234'
    dddd: ['123', '123', '123', '123', '123', '123', '123', '123', '123', '123']
y|  a: '1234'
    ddddd:
        ['123', '123', '123', '123', '123', '123', '123', '123', '123', '123']
y|  e:
        'a
        b'
"""
    )
    a = aa = 2 * ["0123456789ABC"]
    print(l0, file=sys.stderr)
    y(a, line_length=40)
    y(aa, line_length=40)
    y(aa, line_length=41)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
         1         2         3         4         5         6         7         8
.........0.........0.........0.........0.........0.........0.........0.........0
y| a: ['0123456789ABC', '0123456789ABC']
y|  aa:
        ['0123456789ABC',
         '0123456789ABC']
y| aa: ['0123456789ABC', '0123456789ABC']
"""
    )


def test_compact(capsys):
    if PY2:
        return
    a = 9 * ["0123456789"]
    y(a)
    y(a, compact=True)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y|  a:
        ['0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789']
y|  a:
        ['0123456789', '0123456789', '0123456789', '0123456789', '0123456789',
         '0123456789', '0123456789', '0123456789', '0123456789']
"""
    )


def test_depth_indent(capsys):
    s = "=============================================="
    a = [s + "1", [s + "2", [s + "3", [s + "4"]]], s + "1"]
    y(a, indent=4)
    y(a, depth=2, indent=4)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y|  a:
        [   '==============================================1',
            [   '==============================================2',
                [   '==============================================3',
                    ['==============================================4']]],
            '==============================================1']
y|  a:
        [   '==============================================1',
            ['==============================================2', [...]],
            '==============================================1']
"""
    )


def test_enabled(capsys):
    with y.preserve():
        y("One")
        y.configure(enabled=False)
        y("Two")
        s = y("Two", as_str=True)
        assert s == ""
        y.configure(enabled=True)
        y("Three")

    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| 'One'
y| 'Three'
"""
    )


def test_enabled2(capsys):
    with y.preserve():
        y.configure(enabled=False)
        line0 = y("line0")
        noline0 = y(prefix="no0")
        pair0 = y("p0", "p0")
        s0 = y("s0", as_str=True)
        y.configure(enabled=[])
        line1 = y("line1")
        noline1 = y(prefix="no1")
        pair1 = y("p1", "p1")
        s1 = y("s1", as_str=True)
        y.configure(enabled=True)
        line2 = y("line2")
        noline2 = y(prefix="no2")
        pair2 = y("p2", "p2")
        s2 = y("s2", as_str=True)
        out, err = capsys.readouterr()
        assert "line0" not in err and "p0" not in err and "no0" not in err
        assert "line1" not in err and "p1" not in err and "no1" not in err
        assert "line2" in err and "p2" in err and "no2" in err
        assert line0 == "line0"
        assert line1 == "line1"
        assert line2 == "line2"
        assert noline0 is None
        assert noline1 is None
        assert noline2 is None
        assert pair0 == ("p0", "p0")
        assert pair1 == ("p1", "p1")
        assert pair2 == ("p2", "p2")
        assert s0 == ""
        assert s1 == ""
        assert s2 == "y| 's2'\n"


def test_enabled3(capsys):
    with y.preserve():
        y.configure(enabled=[])
        y(2)
        with pytest.raises(TypeError):

            @y()
            def add2(x):
                return x + 2

        with pytest.raises((AttributeError, TypeError)):
            with y():
                pass

        @y(decorator=True)
        def add2(x):
            return x + 2

        with y(context_manager=True):
            pass


def test_multiple_as():
    with pytest.raises(TypeError):
        y(1, decorator=True, context_manager=True)
    with pytest.raises(TypeError):
        y(1, decorator=True, as_str=True)
    with pytest.raises(TypeError):
        y(1, context_manager=True, as_str=True)


def test_wrap_indent():
    s = 4 * ["*******************"]
    res = y(s, compact=True, as_str=True)
    assert res.splitlines()[1].startswith("    s")
    res = y(s, compact=True, as_str=True, wrap_indent="....")
    assert res.splitlines()[1].startswith("....s")
    res = y(s, compact=True, as_str=True, wrap_indent=2)
    assert res.splitlines()[1].startswith("  s")
    res = y(s, compact=True, as_str=True, wrap_indent=[])
    assert res.splitlines()[1].startswith("[]s")


def test_traceback(capsys):
    with y.preserve():
        y.show_traceback = True
        y()
        out, err = capsys.readouterr()
        assert err.count("traceback") == 2

        @y
        def p():
            pass

        p()
        out, err = capsys.readouterr()
        assert err.count("traceback") == 2
        with y():
            pass
        out, err = capsys.readouterr()
        assert err.count("traceback") == 2


def test_enforce_line_length(capsys):
    s = 80 * "*"
    y(s)
    y(s, enforce_line_length=True)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y|  s: '********************************************************************************'
y|  s: '************************************************************************
"""
    )
    with y.preserve():
        y.configure(line_length=20, show_line_number=True)
        y()
        out, err1 = capsys.readouterr()
        y(enforce_line_length=True)
        out, err2 = capsys.readouterr()
        err1 = err1.rstrip("\n")
        err2 = err2.rstrip("\n")
        assert len(err2) == 20
        assert err1[10:20] == err2[10:20]
        assert len(err1) > 20
    res = y("abcdefghijklmnopqrstuvwxyz", p="", ell=1, ll=20, as_str=True).rstrip("\n")
    assert res == "'abcdefghijklmnopqrs"
    assert len(res) == 20


def test_check_output(capsys):
    """special Pythonista code, as that does not reload x1 and x2"""
    if "x1" in sys.modules:
        del sys.modules["x1"]
    if "x2" in sys.modules:
        del sys.modules["x2"]
    del sys.modules["ycecream"]
    from ycecream import y

    """ end of special Pythonista code """
    with y.preserve():
        with tempfile.TemporaryDirectory() as tmpdir:
            x1_file = Path(tmpdir) / "x1.py"
            with open(str(x1_file), "w") as f:
                print(
                    """\
def check_output():
    from ycecream import y
    import x2

    y.configure(show_line_number=True, show_exit= False)
    x2.test()
    y(1)
    y(
    1
    )
    with y(prefix="==>"):
        y()

    with y(



        prefix="==>"

        ):
        y()

    @y
    def x(a, b=1):
        pass
    x(2)

    @y()




    def x(


    ):
        pass

    x()
""",
                    file=f,
                )

            x2_file = Path(tmpdir) / "x2.py"
            with open(str(x2_file), "w") as f:
                print(
                    """\
from ycecream import y

def test():
    @y()
    def myself(x):
        y(x)
        return x

    myself(6)
    with y():
        pass
""",
                    file=f,
                )
            sys.path = [tmpdir] + sys.path
            import x1

            x1.check_output()
            sys.path.pop(0)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| #5[x2.py] in test() ==> called myself(6)
y| #6[x2.py] in test.myself() ==> x: 6
y| #10[x2.py] in test() ==> enter
y| #7[x1.py] in check_output() ==> 1
y| #8[x1.py] in check_output() ==> 1
==>#11[x1.py] in check_output() ==> enter
y| #12[x1.py] in check_output()
==>#14[x1.py] in check_output() ==> enter
y| #21[x1.py] in check_output()
y| #24[x1.py] in check_output() ==> called x(2)
y| #33[x1.py] in check_output() ==> called x()
"""
    )


def test_provided(capsys):
    with y.preserve():
        y("1")
        y("2", provided=True)
        y("3", provided=False)
        y.enabled = False
        y("4")
        y("5", provided=True)
        y("6", provided=False)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| '1'
y| '2'
"""
    )


def test_assert_():
    y.assert_(True)
    with pytest.raises(AssertionError):
        y.assert_(False)

    with y.preserve():
        y.enabled = False
        y.assert_(True)
        y.assert_(False)


def test_propagation():
    with y.preserve():
        y0 = y.fork()
        y1 = y0.fork()
        y.p = "x"
        y2 = y.clone()

        assert y.p == "x"
        assert y0.p == "x"
        assert y1.p == "x"
        assert y2.p == "x"

        y1.p = "xx"
        assert y.p == "x"
        assert y0.p == "x"
        assert y1.p == "xx"
        assert y2.p == "x"

        y1.p = None
        assert y.p == "x"
        assert y0.p == "x"
        assert y1.p == "x"
        assert y2.p == "x"

        y.p = None
        assert y.p == "y| "
        assert y0.p == "y| "
        assert y1.p == "y| "
        assert y2.p == "x"


def test_delta_propagation():
    with y.preserve():
        y_delta_start = y.delta
        y0 = y.fork()
        y1 = y0.fork()
        y.dl = 100
        y2 = y.clone()

        assert 100 < y.delta < 110
        assert 100 < y0.delta < 110
        assert 100 < y1.delta < 110
        assert 100 < y2.delta < 110

        y1.delta = 200
        assert 100 < y.delta < 110
        assert 100 < y0.delta < 110
        assert 200 < y1.delta < 210
        assert 100 < y2.delta < 110

        y1.delta = None
        assert 100 < y.delta < 110
        assert 100 < y0.delta < 110
        assert 100 < y1.delta < 110
        assert 100 < y2.delta < 110

        y.delta = None
        assert 0 < y.delta < y_delta_start + 10
        assert 0 < y0.delta < y_delta_start + 10
        assert 0 < y1.delta < y_delta_start + 10
        assert 100 < y2.delta < 110


def test_separator(capsys):
    a = 12
    b = 4 * ["test"]
    y(a, b)
    y(a, b, sep="")
    y(a, separator="")
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| a: 12, b: ['test', 'test', 'test', 'test']
y|  a: 12
    b: ['test', 'test', 'test', 'test']
y| a: 12
"""
    )


def test_equals_separator(capsys):
    a = 12
    b = 4 * ["test"]
    y(a, b)
    y(a, b, equals_separator=" ==> ")
    y(a, b, es=" = ")

    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| a: 12, b: ['test', 'test', 'test', 'test']
y| a ==> 12, b ==> ['test', 'test', 'test', 'test']
y| a = 12, b = ['test', 'test', 'test', 'test']
"""
    )


def test_context_separator(capsys):
    a = 12
    b = 2 * ["test"]
    y(a, b, show_line_number=True)
    y(a, b, sln=1, context_separator=" ... ")

    out, err = capsys.readouterr()
    lines = err.split("\n")
    assert lines[0].endswith(" ==> a: 12, b: ['test', 'test']")
    assert lines[1].endswith(" ... a: 12, b: ['test', 'test']")


def test_wrap_indent(capsys):
    with y.preserve():
        y.separator = ""
        y(1, 2)
        y(1, 2, prefix="yy| ")
        y(1, 2, prefix="yyy| ")
        y.wrap_indent = "...."
        y(1, 2, prefix="yy| ")
        y(1, 2, prefix="yyy| ")
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y|  1
    2
yy| 1
    2
yyy|
    1
    2
yy| 1
....2
yyy|
....1
....2
"""
    )


@pytest.mark.skipif(sys.version_info < (3, 6), reason="f-strings require Python >= 3.6")
def test_fstrings(capsys):
    test_code = """\
hello='world'
with y.preserve():
    y('hello, world')
    y(hello)
    y(f'hello={hello}')

with y.preserve():
    y.values_only = True
    y('hello, world')
    y(hello)
    y(f'hello={hello}')

with y.preserve():
    y.values_only_for_fstrings=True
    y('hello, world')
    y(hello)
    y(f'hello={hello}')

with y.preserve():
    y.voff=True
    y.vo=True
    y('hello, world')
    y(hello)
    y(f'hello={hello}')"""
    exec(test_code)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| 'hello, world'
y| 'world'
y| 'hello=world'
y| 'hello, world'
y| 'world'
y| 'hello=world'
y| 'hello, world'
y| 'world'
y| 'hello=world'
y| 'hello, world'
y| 'world'
y| 'hello=world'
"""
    )


if __name__ == "__main__":
    pytest.main(["-vv", "-s", "-x", __file__])
