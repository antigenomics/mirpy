d = {'a1': 1, 'a2': 0, 'a3' : 3, 'b1' : 2}

for ((g1, s1), (g2, s2)) in zip(d.items(), d.items()):
    print((g1, g2), (s1, s2))

print(d)

print((x, 1) for x in d.keys())
print(dict((x, 1) for x in d.keys()))