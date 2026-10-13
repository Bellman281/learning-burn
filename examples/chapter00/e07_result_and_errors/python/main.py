def parse_sum(a, b):
    return int(a) + int(b)       # raises ValueError

try:
    print(parse_sum("2", "40"))  # 42
    print(parse_sum("2", "x"))   # boom
except ValueError as e:
    print("bad:", e)

# nothing in the signature warned you it could raise
