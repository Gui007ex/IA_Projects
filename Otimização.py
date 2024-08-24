import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def decor_plot(sub):
    sub.legend()
    sub.set_xlim(0, 1)
    sub.set_ylim(0, 1)
    sub.set_zlim(0, 1)
    sub.set_xlabel('X')
    sub.set_ylabel('Y')
    sub.set_zlabel('Z')

def plotar(p1,p2,sub,color="black"):
    return sub.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],color)

def perturb(x):
    i1,i2 = np.random.permutation(len(x))[0:2]
    x[i1], x[i2] = x[i2], x[i1]
    return x

def f(cidades,x):
    s = 0
    for i in range(len(x)):
        p1 = cidades[x[i]]
        p2 = cidades[x[(i+1)%len(x)]]
        s += np.linalg.norm(p1-p2)
    return s

def plot_inicial(cidades,x,pos,grid):
    ax = fig.add_subplot(grid,grid,pos,projection="3d")
    decor_plot(ax)
    ax.scatter(cidades[:,0], cidades[:,1], cidades[:,2])
    lines = []

    for i in range(len(x)):
        p1 = cidades[x[i]]
        p2 = cidades[x[(i+1)%len(x)]]
        if i == 0:
            l = plotar(p1,p2,ax,color="green")
        elif i+1 == len(x):
            l = plotar(p1,p2,ax,color="red")
        else:
            l = plotar(p1,p2,ax)
        lines.append(l[0])

    return lines,ax

def atualizar_plot(cidades,x,lines,ax):
    for line in lines:
        line.remove()
    
    for i in range(len(x)):
        p1 = cidades[x[i]]
        p2 = cidades[x[(i+1)%len(x)]]
        if i == 0:
            l = plotar(p1,p2,ax,color="green")
        elif i+1 == len(x):
            l = plotar(p1,p2,ax,color="red")
        else:
            l = plotar(p1,p2,ax)
        lines[i] = l[0]

fig = plt.figure(figsize=(10,8))
grid = 2
executions = grid**2
plots = {}
p = 15
cidades = np.random.rand(p, 3)

for i in range(executions):
    plots[i] = {}
    plots[i]["x_opt"] = np.random.permutation(p)
    plots[i]["f_opt"] = f(cidades,plots[i]["x_opt"])
    plots[i]["lines"],plots[i]["ax"] = plot_inicial(cidades,plots[i]["x_opt"],i+1,grid)

max_it = 10000
for i in range(max_it):
    for k in plots:
        x_cand = perturb(np.copy(plots[k]["x_opt"]))
        f_cand = f(cidades,x_cand)
        if f_cand < plots[k]["f_opt"]:
            plt.pause(.025)
            plots[k]["x_opt"] = x_cand
            plots[k]["f_opt"] = f_cand
            red_patch = mpatches.Patch(color='black', label=f"f_opt: {plots[k]["f_opt"]}")
            plots[k]["ax"].legend(handles=[red_patch])
            atualizar_plot(cidades,plots[k]["x_opt"],plots[k]["lines"],plots[k]["ax"])

plt.show()