a = [1, 2, 3]
b = a              # b and a are the SAME list
a.append(4)
print(b)           # [1, 2, 3, 4] -- b saw it too

import copy
c = copy.copy(a)   # want a real copy? ask

n = 10
m = n              # ints are immutable; no aliasing worry
