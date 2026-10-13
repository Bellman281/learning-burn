def largest(xs):
    return max(xs)            # works on anything comparable

print(largest([3, 7, 2, 9, 4]))      # 9
print(largest([1.5, 0.5, 2.25]))     # 2.25
print(largest(["burn", "rust"]))     # "rust"

# type hints are optional, and not enforced at runtime
