import numpy as np
ROWS, COLS = 5, 5
GOAL = (4, 4)
GAMMA = 0.9
V = np.zeros((ROWS, COLS))
for _ in range(50):
    newV = V.copy()
    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) == GOAL:
                continue
            moves = []
            if i > 0:
                moves.append(V[i-1][j])
            if i < ROWS-1:
                moves.append(V[i+1][j])
            if j > 0:
                moves.append(V[i][j-1])
            if j < COLS-1:
                moves.append(V[i][j+1])
            newV[i][j] = -1 + GAMMA * max(moves)
    V = newV
print("Warehouse Value Function\n")
print(np.round(V,2))
print("\nBest Route")
r, c = 0, 0
print((r,c), end=" ")
while (r,c) != GOAL:
    if c < GOAL[1]:
        c += 1
    else:
        r += 1
    print("->", (r,c), end=" ")


Output:
Warehouse Value Function

[[1.81 3.12 4.58 6.2 8. ]
 [3.12 4.58 6.2 8. 10.]
 [4.58 6.2 8. 10. 12.]
 [6.2 8. 10. 12. 14.]
 [8. 10. 12. 14. 0.]]

Best Route
(0,0) -> (0,1) -> (0,2) -> (0,3) -> (0,4) -> (1,4) -> (2,4) -> (3,4) -> (4,4)
