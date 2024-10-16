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
