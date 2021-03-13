import pytest
from pathlib import Path
import sys
import tempfile

context_start = "y| #"

# make a temporary dict and put a dummy ycecream.json file there, to prevent reading any ycecream.json
with tempfile.TemporaryDirectory() as tmpdir:
    json_filename = Path(tmpdir) / "ycecream.json"
    with open(json_filename, "w") as f:
        print("{}", file=f)
    sys.path = [tmpdir] + sys.path
    from ycecream import y

    sys.path.pop(0)

import ycecream
import datetime
import time
import pytest

FAKE_TIME = datetime.datetime(2021, 1, 1, 0, 0, 0)


@pytest.fixture
def patch_datetime_now(monkeypatch):
    class mydatetime:
        @classmethod
        def now(cls):
            return FAKE_TIME

    monkeypatch.setattr(datetime, "datetime", mydatetime)


@pytest.fixture
def patch_perf_counter(monkeypatch):
    def myperf_counter():
        return 0

    monkeypatch.setattr(time, "perf_counter", myperf_counter)


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
    out, err = capsys.readouterr()
    assert err == "y| hello: 'world'\n"
    assert result == hello


def test_two_arguments(capsys):
    hello = "world"
    l = [1, 2, 3]
    result = y(hello, l)
    out, err = capsys.readouterr()
    assert err == "y| hello: 'world', l: [1, 2, 3]\n"
    assert result == (hello, l)


def test_in_function(capsys):
    def hello(val):
        y(val, show_line_number=True)

    hello("world")
    out, err = capsys.readouterr()
    assert err.startswith(context_start)
    assert err.endswith(" in hello() ==> val: 'world'\n")


def test_prefix(capsys):
    hello = "world"
    y(hello, prefix="==> ")
    out, err = capsys.readouterr()
    assert err == "==> hello: 'world'\n"


def test_dynamic_prefix(capsys):
    i = 0

    def prefix():
        nonlocal i
        i += 1
        return f"{i})"

    hello = "world"
    y(hello, prefix=prefix)
    y(hello, prefix=prefix)
    out, err = capsys.readouterr()
    assert err == "1)hello: 'world'\n2)hello: 'world'\n"

def test_calls():
    with pytest.raises(TypeError):
        ycecream.Y(a=1)
    with pytest.raises(TypeError):
        y.clone(a=1)
    with pytest.raises(TypeError):
        y.configure(a=1)     
    with pytest.raises(TypeError):
        y(12, a=1)
    with pytest.raises(TypeError):
        y(a=1)        
                   
def test_output(capsys, tmpdir):
    result = ""

    def my_output(s):
        nonlocal result
        result += s + "\n"

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

    path = Path(tmpdir) / "x0"
    y(hello, output=path)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    with open(path, "r") as f:
        assert f.read() == "y| hello: 'world'\n"

    path = Path(tmpdir) / "x1"
    y(hello, output=str(path))
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    with open(path, "r") as f:
        assert f.read() == "y| hello: 'world'\n"

    path = Path(tmpdir) / "x2"
    with open(path, "a+") as f:
        y(hello, output=f)
    with pytest.raises(ValueError):  # closed file
        y(hello, output=f)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    with open(path, "r") as f:
        assert f.read() == "y| hello: 'world'\n"

    with pytest.raises(ValueError):
        y(hello, output=1)

    y(hello, output=my_output)
    y(1, output=my_output)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    assert result == "y| hello: 'world'\ny| 1\n"


def test_serialize(capsys):
    def serialize(s):
        return f"{repr(s)} [len={len(s)}]"

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


def test_clone():
    hello = "world"
    z = y.clone(prefix="z| ")
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
    assert s0 == s1 == "y| world: {'EN': 'world', 'NL': 'wereld', 'FR': 'monde', 'DE': 'Welt'}\n"
    assert s2 == "y| world: {'DE': 'Welt', 'EN': 'world', 'FR': 'monde', 'NL': 'wereld'}\n"


def test_multiline():
    a = 1
    b = 2
    l = list(range(15))
    # fmt: off
    s=y((a, b),
        [l,
        l], as_str=True)
    # fmt: on
    assert (
        s
        == """\
y|
    (a, b): (1, 2)
    [l,
    l]:
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
"""
    )

    lines = "\n".join(f"line{i}" for i in range(4))
    result = y(lines, as_str=True)
    assert (
        result
        == """\
y|
    lines:
        'line0
        line1
        line2
        line3'
"""
    )


def test_decorator(capsys, patch_perf_counter):
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
        return x ** y

    assert mul(2, 3) == 2 * 3
    assert div(10, 2) == 10 / 2
    assert add(2, 3) == 2 + 3
    assert sub(10, 2) == 10 - 2
    assert pow(10, 2) == 10 ** 2
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


def test_decorator_edge_cases(capsys, patch_perf_counter):
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
            return f"{self.__class__.__name__}({self.value})"

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


def test_context_manager(capsys, patch_perf_counter):
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


def test_json_reading(tmpdir):
    json_filename = Path(tmpdir) / "ycecream.json"
    with open(json_filename, "w") as f:
        print('{"prefix": "xxx"}', file=f)

    sys.path = [tmpdir] + sys.path
    ycecream.set_defaults()
    sys.path.pop(0)

    y1 = ycecream.Y()

    s = y1(3, as_str=True)
    assert s == "xxx3\n"

    with open(json_filename, "w") as f:
        print('{"prefix1": "xxx"}', file=f)

    sys.path = [tmpdir] + sys.path
    with pytest.raises(ValueError):
        ycecream.set_defaults()
    sys.path.pop(0)

    with open(json_filename, "w") as f:
        print('{"serialize": "xxx"}', file=f)

    sys.path = [tmpdir] + sys.path
    with pytest.raises(ValueError):
        ycecream.set_defaults()
    sys.path.pop(0)

    tmpdir = Path(tmpdir) / "ycecream"
    tmpdir.mkdir()
    json_filename = Path(tmpdir) / "ycecream.json"
    with open(json_filename, "w") as f:
        print('{"prefix": "yyy"}', file=f)

    sys.path = [tmpdir] + sys.path
    ycecream.set_defaults()
    sys.path.pop(0)

    y1 = ycecream.Y()

    s = y1(3, as_str=True)
    assert s == "yyy3\n"
    sys.modules["_ycecream_ignore_json_"] = True  # indicator for ycecream to not read from ycecream.json
    ycecream.set_defaults()


def test_wrapping(capsys):
    l0 = "".join(f"         {c}" for c in "12345678") + "\n" + "".join(f".........0" for c in "12345678")

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
y|
    cccc:
        ['12345678123456789012', '12345678123456789012', '12345678123456789012']
y|
    ccc0:
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
y|
    a: '1234'
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
y|
    a: '1234'
    dddd: ['123', '123', '123', '123', '123', '123', '123', '123', '123', '123']
y|
    a: '1234'
    ddddd:
        ['123', '123', '123', '123', '123', '123', '123', '123', '123', '123']
y|
    e:
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
y|
    aa:
        ['0123456789ABC',
         '0123456789ABC']
y| aa: ['0123456789ABC', '0123456789ABC']
"""
    )


def test_compact(capsys):
    a = 9 * ["0123456789"]
    y(a)
    y(a, compact=True)
    out, err = capsys.readouterr()
    assert (
        err
        == """\
y|
    a:
        ['0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789',
         '0123456789']
y|
    a:
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
y|
    a:
        [   '==============================================1',
            [   '==============================================2',
                [   '==============================================3',
                    ['==============================================4']]],
            '==============================================1']
y|
    a:
        [   '==============================================1',
            ['==============================================2', [...]],
            '==============================================1']
"""
    )


def test_enable(capsys):
    with y.preserve():
        y("One")
        y.configure(enabled=False)
        y("Two")
        s = y("Two", as_str=True)
        assert s == "y| 'Two'\n"
        y.configure(enabled=True)
        y("Three")

        ycecream.enable(False)
        y("Four")
        ycecream.enable(True)
        y("Five")

    out, err = capsys.readouterr()
    assert (
        err
        == """\
y| 'One'
y| 'Three'
y| 'Five'
"""
    )


def test_check_output(capsys):
    with y.preserve():
        with tempfile.TemporaryDirectory() as tmpdir:
            x1_file = Path(tmpdir) / "x1.py"
            with open(x1_file, "w") as f:
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
            with open(x2_file, "w") as f:
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
y| #6[x2.py] in myself() ==> x: 6
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


if __name__ == "__main__":
    pytest.main(["-vv", "-s", "-x", __file__])
