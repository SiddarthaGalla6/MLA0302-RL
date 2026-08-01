import numpy as np
import csv
import os
from google.colab import files
ACTIONS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
GAMMA=0.9
THETA=1e-4
STATE={'ROWS':0,'COLS':0,'ITEM_LOCATIONS':set(),'OBSTACLES':set(),'GOAL':None,'GRID':None,'V':None}
def load_layout(path):
    with open(path,'r') as f:
        reader=csv.reader(f)
        grid=[row for row in reader if row]
    rows=len(grid)
    cols=len(grid[0])
    items=set()
    obstacles=set()
    goal=None
    for r in range(rows):
        for c in range(cols):
            val=grid[r][c].strip().upper()
            if val=='I':
                items.add((r,c))
            elif val=='X':
                obstacles.add((r,c))
            elif val=='G':
                goal=(r,c)
    STATE['ROWS']=rows
    STATE['COLS']=cols
    STATE['ITEM_LOCATIONS']=items
    STATE['OBSTACLES']=obstacles
    STATE['GOAL']=goal
    STATE['GRID']=grid
    print("Warehouse layout loaded successfully from",path)
def reward(state):
    if state==STATE['GOAL']:
        return 5
    if state in STATE['ITEM_LOCATIONS']:
        return 2
    if state in STATE['OBSTACLES']:
        return -2
    return -0.1
def is_valid(state):
    r,c=state
    return 0<=r<STATE['ROWS'] and 0<=c<STATE['COLS']
def next_state(state,action):
    dr,dc=ACTIONS[action]
    ns=(state[0]+dr,state[1]+dc)
    return ns if is_valid(ns) else state
def fixed_policy(state):
    r,c=state
    goal=STATE['GOAL']
    if c<goal[1]:
        return 'RIGHT'
    if r<goal[0]:
        return 'DOWN'
    return 'RIGHT'
def policy_evaluation():
    rows=STATE['ROWS']
    cols=STATE['COLS']
    goal=STATE['GOAL']
    obstacles=STATE['OBSTACLES']
    V=np.zeros((rows,cols))
    iteration=0
    while True:
        delta=0
        new_V=V.copy()
        for r in range(rows):
            for c in range(cols):
                state=(r,c)
                if state in obstacles or state==goal:
                    continue
                action=fixed_policy(state)
                ns=next_state(state,action)
                new_V[r,c]=reward(ns)+GAMMA*V[ns]
                delta=max(delta,abs(new_V[r,c]-V[r,c]))
        V=new_V
        iteration+=1
        if delta<THETA:
            break
    STATE['V']=V
    print("Policy evaluation completed in",iteration,"iterations")
    return V,iteration
def display_layout():
    if STATE['GRID'] is None:
        print("No layout loaded. Please load a CSV file first.")
        return
    print("Warehouse Layout")
    for r in range(STATE['ROWS']):
        row=[]
        for c in range(STATE['COLS']):
            cell=(r,c)
            if cell==STATE['GOAL']:
                row.append('G')
            elif cell in STATE['ITEM_LOCATIONS']:
                row.append('I')
            elif cell in STATE['OBSTACLES']:
                row.append('X')
            else:
                row.append('.')
        print(' '.join(row))
def display_value_function():
    if STATE['V'] is None:
        print("Value function not computed yet. Please run policy evaluation first.")
        return
    print("Value Function")
    print(np.round(STATE['V'],2))
def save_value_function(path):
    if STATE['V'] is None:
        print("Value function not computed yet. Please run policy evaluation first.")
        return
    with open(path,'w',newline='') as f:
        writer=csv.writer(f)
        for r in range(STATE['ROWS']):
            row=[round(STATE['V'][r,c],2) for c in range(STATE['COLS'])]
            writer.writerow(row)
    print("Value function saved successfully to",path)
def upload_csv():
    print("Please choose a CSV file to upload...")
    uploaded=files.upload()
    if not uploaded:
        return None
    return list(uploaded.keys())[0]
def download_csv(path):
    files.download(path)
def menu():
    while True:
        print("\n===== Warehouse Robot Policy Evaluation Menu =====")
        print("1. Load Warehouse Layout from CSV")
        print("2. Display Warehouse Layout")
        print("3. Run Policy Evaluation")
        print("4. Display Value Function")
        print("5. Save Value Function to CSV")
        print("6. Exit")
        choice=input("Enter your choice (1-6): ").strip()
        if choice=='1':
            path=upload_csv()
            if path and os.path.exists(path):
                load_layout(path)
            else:
                print("No file uploaded. Please try again.")
        elif choice=='2':
            display_layout()
        elif choice=='3':
            if STATE['GRID'] is None:
                print("Please load a warehouse layout first.")
            else:
                policy_evaluation()
        elif choice=='4':
            display_value_function()
        elif choice=='5':
            name=input("Enter output CSV file name (e.g. value_function.csv): ").strip()
            if not name:
                name="value_function.csv"
            if not name.endswith(".csv"):
                name=name+".csv"
            save_value_function(name)
            download_csv(name)
        elif choice=='6':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")
if __name__=="__main__":
    menu()


Output:
===== Warehouse Robot Policy Evaluation Menu =====
1. Load Warehouse Layout from CSV
2. Display Warehouse Layout
3. Run Policy Evaluation
4. Display Value Function
5. Save Value Function to CSV
6. Exit
Enter your choice (1-6): 3
Policy evaluation completed in 9 iterations

===== Warehouse Robot Policy Evaluation Menu =====
1. Load Warehouse Layout from CSV
2. Display Warehouse Layout
3. Run Policy Evaluation
4. Display Value Function
5. Save Value Function to CSV
6. Exit
Enter your choice (1-6): 4
Value Function
[[ 1.87  2.19  2.54  2.94  3.37]
 [ 0.29 -1.9  -2.    0.    3.86]
 [-1.9  -2.    0.    3.86  4.4 ]
 [-1.81 -1.9  -2.    0.    5.  ]
