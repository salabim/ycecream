from ycecream import y
y.show_line_number = True
y.prefix="..."
y.output="stdout"
y.enabled=False
l = [1, 1, 0, -1, 2, 3, 1, 1, 4, 5, 6, 4, 0, 3, 1, 0]
for el in l[:]:
    y(el, l)    
    if el <= 1:
        y(el,l)
        l.remove(el)
print(l)
        