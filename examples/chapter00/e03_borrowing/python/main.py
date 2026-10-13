def char_count(text):
    return len(text)      # only reads it

def shout(text):
    return text.upper() + "!"

s = "burn"
n = char_count(s)         # s is still usable
print(s, n)               # burn 4

# a function CAN mutate a list you pass it --
# silently, with nothing in the signature to warn you
