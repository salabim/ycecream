from ycecream import y

@y(show_enter=False)
def do_sort(i):
    n = 10 ** i
    x = sorted(list(range(n)))
    return f"{n:9d}" 

for i in range(7):
    do_sort(i)

