import numpy as np
GRID_SIZE = 5
ACTIONS = ["Up", "Down", "Left", "Right"]
MOVE = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1)
}
GAMMA = 0.9
STEP_COST = -0.04
DIRT_REWARD = 1
OBSTACLE_PENALTY = -1
grid = [
    ["S", ".", "D", "#", "."],
    [".", "#", ".", ".", "D"],
    ["D", ".", "D", "#", "."],
    [".", "#", ".", "D", "."],
    [".", "D", ".", ".", "D"]
]
start = (0, 0)
def valid(pos):
    r, c = pos
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE
def move(pos, action):
    dr, dc = MOVE[action]
    nr = pos[0] + dr
    nc = pos[1] + dc
    if not valid((nr, nc)):
        return pos
    if grid[nr][nc] == "#":
        return pos
    return (nr, nc)
def reward(pos):
    r, c = pos
    if grid[r][c] == "D":
        return DIRT_REWARD
    return STEP_COST
V = np.zeros((GRID_SIZE, GRID_SIZE))
policy = [["" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
while True:
    delta = 0
    newV = V.copy()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == "#":
                continue
            values = []
            for action in ACTIONS:
                nr, nc = move((r, c), action)
                value = reward((nr, nc)) + GAMMA * V[nr][nc]
                values.append((value, action))
            best_value, best_action = max(values)
            newV[r][c] = best_value
            policy[r][c] = best_action
            delta = max(delta, abs(best_value - V[r][c]))
    V = newV
    if delta < 0.001:
        break
arrow = {
    "Up": "^",
    "Down": "v",
    "Left": "<",
    "Right": ">"
}
print("\nOptimal Policy\n")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        if grid[r][c] == "#":
            print("#", end=" ")
        elif grid[r][c] == "S":
            print("S", end=" ")
        else:
            print(arrow[policy[r][c]], end=" ")
    print()
print("\nRobot Movement\n")
position = start
visited = set()
steps = 0
total_reward = 0
while steps < 20:
    r, c = position
    print("Step", steps,
          " Position:", position,
          " Cell:", grid[r][c])
    if grid[r][c] == "D" and position not in visited:
        total_reward += DIRT_REWARD
        visited.add(position)
        print(" Dirt Cleaned")
    action = policy[r][c]
    position = move(position, action)
    total_reward += STEP_COST
    steps += 1
print("\nTotal Reward =", total_reward)


Output :
Optimal Policy
S > ^ # v 
v # ^ > > 
< > > # ^ 
^ # ^ ^ < 
> ^ < ^ > 

Robot Movement : 
Step 0  Position: (0, 0)  Cell: S
Step 1  Position: (0, 1)  Cell: .
Step 2  Position: (0, 2)  Cell: D
 Dirt Cleaned
Step 3  Position: (0, 2)  Cell: D
Step 4  Position: (0, 2)  Cell: D
Step 5  Position: (0, 2)  Cell: D
Step 6  Position: (0, 2)  Cell: D
Step 7  Position: (0, 2)  Cell: D
Step 8  Position: (0, 2)  Cell: D
Step 9  Position: (0, 2)  Cell: D
Step 10  Position: (0, 2)  Cell: D
Step 11  Position: (0, 2)  Cell: D
Step 12  Position: (0, 2)  Cell: D
Step 13  Position: (0, 2)  Cell: D
Step 14  Position: (0, 2)  Cell: D
Step 15  Position: (0, 2)  Cell: D
Step 16  Position: (0, 2)  Cell: D
Step 17  Position: (0, 2)  Cell: D
Step 18  Position: (0, 2)  Cell: D
Step 19  Position: (0, 2)  Cell: D

Total Reward = 0.1999999999999998
