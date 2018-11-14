import numpy as np

def Df(x,y):
    '''return gradient and hessian'''
    return np.array([
        1 / (1-x) - 1.75 / x - 1 / (1 + x - y),
        1 / (1-y) - 1/y + 1 / (1 + x - y),
    ]), np.array([
        [1/(1-x)**2 + 1.75 / x**2 + 1/(1 + x - y)**2,-1 / (1 + x - y)**2],
        [-1 / (1 + x - y)**2, 1/(1-y)**2 + 1 / y**2 + 1 / (1 + x - y)**2],
    ])

xys = [
    np.array([0.2,0.7]),
    np.array([0.25,0.9]),
    np.array([0.3,0.1]),
]

for xy in xys: # do newton
    xy_ = xy.copy()
    print(r'\draw[red!40, thick] ',end='')
    for _ in range(100):
        print('({:0.8f},{:0.8f}) -- '.format(xy[0],xy[1]),end='')
        grad, Hf = Df(xy[0], xy[1])
        xy = xy - 0.1 * np.linalg.solve(Hf, grad)
    print('({:0.8f},{:0.8f});'.format(xy[0],xy[1]))

print('\n\nFinding Optimal:')
xy = np.array([0.5,0.5])
for _ in range(100):
    grad, Hf = Df(xy[0], xy[1])
    xy = xy - np.linalg.solve(Hf, grad)
print('({:0.8f},{:0.8f})'.format(xy[0],xy[1]))
