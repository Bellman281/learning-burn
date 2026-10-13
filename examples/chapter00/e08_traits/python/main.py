class Circle:
    def area(self): ...

class Square:
    def area(self): ...

# duck typing: anything with .area() will do
def describe(shape):
    print(shape.area())

def total_area(shapes):
    return sum(s.area() for s in shapes)

# the contract exists only in your head
