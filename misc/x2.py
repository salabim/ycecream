from ycecream import y

def test():
    @y()
    def myself(x):
        y(x)
        return x
        
    myself(6)
    with y():
        pass        

