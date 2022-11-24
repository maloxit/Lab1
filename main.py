import intvalpy as intv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


delta = 0.03

mat = np.matrix([[1.05, 1], [0.95, 1]])
delta_mat = np.matrix([[delta, delta], [delta, delta]])

mid_int_mat = intv.Interval(mat, mat)
delta_int_mat = intv.Interval(-delta_mat, delta_mat)
A = mid_int_mat + delta_int_mat

print(mid_int_mat)
print(delta_int_mat)
print(A)

det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

print(det)

for i00 in (-delta, delta):
    for i01 in (-delta, delta):
        for i10 in (-delta, delta):
            for i11 in (-delta, delta):
                det = (mat[0, 0] + i00) * (mat[1, 1] + i11) - (mat[0, 1] + i01) * (mat[1, 0] + i10)
                print(det)


def booth(x,y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2



def himmelbou(x,y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def optim(func, start, eps):
    boxes = [[intv.Interval(*start[0]), intv.Interval(*start[1]), None]]
    boxes[0][2] = func(boxes[0][0], boxes[0][1]).a
    trace = []
    while True:
        boxes.sort(key=lambda tup : tup[2])
        box = boxes[0]
        trace.append([box[0].mid, box[1].mid])
        if box[0].wid < eps and box[1].wid < eps:
            break
        boxes = boxes[1:]
        if box[0].wid > box[1].wid:
            box1 = [intv.Interval(box[0].a, box[0].mid), intv.Interval(box[1].a, box[1].b), None]
            box1[2] = func(box1[0], box1[1]).a
            box2 = [intv.Interval(box[0].mid, box[0].b), intv.Interval(box[1].a, box[1].b), None]
            box2[2] = func(box2[0], box2[1]).a
            boxes.append(box1)
            boxes.append(box2)
        else:
            box1 = [intv.Interval(box[0].a, box[0].b), intv.Interval(box[1].a, box[1].mid), None]
            box1[2] = func(box1[0], box1[1]).a
            box2 = [intv.Interval(box[0].a, box[0].b), intv.Interval(box[1].mid, box[1].b), None]
            box2[2] = func(box2[0], box2[1]).a
            boxes.append(box1)
            boxes.append(box2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(*start[0], 0.05)
    y = np.arange(*start[1], 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(func(np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)

    ax.contour(X, Y, Z, 20)

    mids = np.zeros([2, len(boxes)])

    for i in range(len(boxes)):
        mids[0, i] = boxes[i][0].mid
        mids[1, i] = boxes[i][1].mid
        ax.add_patch(Rectangle((boxes[i][0].a, boxes[i][1].a), boxes[i][0].wid, boxes[i][1].wid, fill=False, edgecolor = 'blue', alpha=0.5))

    trace = np.array(trace)
    ax.plot(trace[:,0], trace[:,1], 'g--', alpha=0.9, linewidth=0.5)

    z_mids = np.array(func(mids[0], mids[1]))
    ax.scatter(mids[0], mids[1], c=z_mids, marker='+')


optim(booth, [[-10, 10], [-10, 10]], 0.01)
optim(himmelbou, [[-5, 5], [-5, 5]], 0.01)
plt.show()