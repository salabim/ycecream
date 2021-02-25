import pytest
from pathlib import Path
import sys

context_start = "y| " + Path(__file__).name + ":"
from ycecream import y


def test_no_arguments(capsys):
    result = y()
    out, err = capsys.readouterr()
    assert err.startswith(context_start)
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


def test_output_to_stdout(capsys, tmpdir):
    result = ''
    def my_output(s):
        nonlocal result
        result += s + '\n'
        
    
    hello = "world"
    y(hello, output=print)
    out, err = capsys.readouterr()
    assert out == "y| hello: 'world'\n"
    assert err == ""
    y(hello, output=sys.stdout)
    out, err = capsys.readouterr()
    assert out == "y| hello: 'world'\n"
    assert err == ""
    y(hello, output='')
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    
    y(hello, output=print)
    out, err = capsys.readouterr()
    assert out == "y| hello: 'world'\n"
    assert err == ""    
    
    path = Path(tmpdir) / 'x0'
    y(hello, output=path)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    with open(path, 'r') as f:
        assert f.read() == "y| hello: 'world'\n" 
        
    path = Path(tmpdir) / 'x1'
    y(hello, output=str(path))
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    with open(path, 'r') as f:
        assert f.read() == "y| hello: 'world'\n"      
        
    path = Path(tmpdir) / 'x2'   
    with open(path, 'a+') as f:                   
        y(hello, output=f)
    with pytest.raises(ValueError):  # closed file
        y(hello, output=f)
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    with open(path, 'r') as f:
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
    assert "\u0394 " in err


def test_as_str(capsys):
    hello = "world"
    s = y(hello, as_str=True)
    y(hello)
    out, err = capsys.readouterr()
    assert err[:-1] == s


def test_clone():
    hello = "world"
    z = y.clone(prefix="z| ")
    sy = y(hello, as_str=True)
    with y.preserve():
        y.configure(show_context=True)
        sz = z(hello, as_str=True)
        assert sy.replace("y", "z") == sz


def test_sort_dicts():
    world = {"EN": "world", "NL": "wereld", "FR": "monde", "DE": "Welt"}
    s0 = y(world, as_str=True)
    s1 = y(world, sort_dicts=False, as_str=True)
    s2 = y(world, sort_dicts=True, as_str=True)
    assert s0 == s1 == "y| world: {'EN': 'world', 'NL': 'wereld', 'FR': 'monde', 'DE': 'Welt'}"
    assert s2 == "y| world: {'DE': 'Welt', 'EN': 'world', 'FR': 'monde', 'NL': 'wereld'}"


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
        == """
y| (a, b): (1, 2)
   [l,
   l]: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]""".strip()
    )

    lines = "\n".join(f"line{i}" for i in range(4))
    result = y(lines, as_str=True)
    assert (
        result
        == """
y| lines: 'line0
           line1
           line2
           line3'""".strip()
    )

def test_decorator(capsys):
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
    @y(show_enter= False, show_exit=False)
    def pos(x, y):
        return x ** y
    assert mul(2,3) == 2 * 3
    assert div(10, 2) == 10 / 2
    assert add(2, 3) == 2 + 3
    assert sub(10, 2) == 10 - 2
    assert pow(10, 2) == 10 ** 2
    out, err = capsys.readouterr()
    assert err == '''y| called mul(2, 3)
y| returned 6 from mul(2, 3)
y| called div(10, 2)
y| returned 5.0 from div(10, 2)
y| returned 5 from add(2, 3)
y| called sub(10, 2)
'''

        
    
    
if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
