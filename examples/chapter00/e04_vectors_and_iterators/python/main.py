xs = [1, 2, 3, 4, 5]

squares = [x*x for x in xs if x % 2 == 1]   # [1, 9, 25]

total = sum(xs)      # 15
mx    = max(xs)      # 5

for i, x in enumerate(xs):
    print(f"xs[{i}]={x}", end=" ")

doubled = []
for x in xs:
    doubled.append(x * 2)
