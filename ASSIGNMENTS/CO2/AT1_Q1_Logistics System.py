import random
locations = ["Warehouse", "A", "B", "Customer"]
Q = [0, 0, 0, 0]
learning_rate = 0.5
episodes = 20
for episode in range(episodes):
    current = 0  
    while current != 3:
        next_location = current + 1
        if next_location == 3:
            reward = 100
        else:
            reward = -1
        Q[current] = Q[current] + learning_rate * (
            reward + Q[next_location] - Q[current]
        )
        current = next_location
print("Learned Q-values:", Q)
print("\nBest Delivery Route:")
for location in locations:
    print(location, end=" -> ")

Output :
Learned Q-values: [97.97989845275879, 98.99799823760986, 99.99990463256836, 0]
Best Delivery Route:
Warehouse -> A -> B -> Customer -> 
