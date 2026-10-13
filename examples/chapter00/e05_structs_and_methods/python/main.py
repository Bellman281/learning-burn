class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return (self.x**2 + self.y**2) ** 0.5

    def scale(self, k):
        self.x *= k
        self.y *= k

p = Point(3.0, 4.0)
print(p.norm())     # 5.0
p.scale(2.0)
