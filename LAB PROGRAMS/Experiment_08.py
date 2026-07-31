road = [
["S",".",".",".","."],
[".","X",".","X","."],
[".",".",".",".","."],
["X",".","X",".","."],
[".",".",".",".","G"]
]
r, c = 0, 0
goal = (4,4)
path = [(r,c)]
while (r,c) != goal:
    if c < 4 and road[r][c+1] != "X":
        c += 1
    elif r < 4 and road[r+1][c] != "X":
        r += 1
    else:
        break
    path.append((r,c))
print("Road Map\n")
for row in road:
    print(row)
print("\nCar Route")
for p in path:
    print(p)
print("\nDestination Reached")

Output:
Road Map

['S', '.', '.', '.', '.']
['.', 'X', '.', 'X', '.']
['.', '.', '.', '.', '.']
['X', '.', 'X', '.', '.']
['.', '.', '.', '.', 'G']

Car Route
(0,0)
(0,1)
(0,2)
(0,3)
(0,4)
(1,4)
(2,4)
(3,4)
(4,4)

Destination Reached
