def first(xs):
    return xs[0] if xs else None

x = first([10, 20, 30])
if x is None:          # you can just... forget this
    print("empty")     # and Python will not stop you
else:
    print("first =", x)

y = first([]) or -1    # -1
