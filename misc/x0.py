import x1
from ycecream import yc
print(dir(x1))
x1.check_output()
class X():
    def x(self):
        yc("test")
X().x()

def a():
    def hello(val):
        yc(val, show_line_number=True)

    hello("world")
a()

from icecream import ic
ic.configureOutput(includeContext=True)
class X():
    def x(self):
        ic("test")
X().x()

def a():
    def hello(val):
        ic(val)

    hello("world")
a()

