import numpy as np
p=[0.2,0.4,0.6]
n=500
def eg():
    c=[0]*3
    v=[0]*3
    r=0
    for i in range(n):
        a=np.random.randint(3) if np.random.rand()<0.1 else np.argmax(v)
        x=1 if np.random.rand()<p[a] else 0
        c[a]+=1
        v[a]+=((x-v[a])/c[a])
        r+=x
    return r
def ucb():
    c=[1,1,1]
    v=[0,0,0]
    r=0
    for i in range(3,n):
        a=np.argmax([v[j]+np.sqrt(np.log(i+1)/c[j]) for j in range(3)])
        x=1 if np.random.rand()<p[a] else 0
        c[a]+=1
        v[a]+=((x-v[a])/c[a])
        r+=x
    return r
def ts():
    a=[1]*3
    b=[1]*3
    r=0
    for i in range(n):
        arm=np.argmax([np.random.beta(a[j],b[j]) for j in range(3)])
        x=1 if np.random.rand()<p[arm] else 0
        a[arm]+=x
        b[arm]+=1-x
        r+=x
    return r
print("Epsilon:",eg())
print("UCB:",ucb())
print("Thompson:",ts())

Output:
Epsilon: 271
UCB: 293
Thompson: 287
