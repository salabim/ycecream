from ycecream_not_embedded import y

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
    y(f'hello={hello}')

print("+++")    

x = 12.3
y(f"{x:0.3e}")
y.values_only_for_fstrings = True
y(f"{x:0.3e}")
