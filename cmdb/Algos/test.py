import numpy as np
import matplotlib.pyplot as plt
import pickle
from instanceGeneration import Instances
 
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


# N = 8#假设有8个点
# x=[1,2,3,4,5,6,7,8]
# y=[1,2,3,4,5,6,7,8]
# c=[1,2,2,2,3,3,4,4]#有4个类别，标签分别是1，2，3，4

f = open('E:\VSCodeSpace\PythonWorkspace\djangoProject\mysite0203\cmdb\Algos/10-nodeInstances', 'rb')
ins = pickle.load(f)
x = ins.a_2d_SitesCoordi[:, 0]  # 横坐标数组，大小[0, 1]
y = ins.a_2d_SitesCoordi[:, 1]  # 纵坐标数组
c = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

m = {1: '*', 0: 'o'}
cm = list(map(lambda x: m[x], c))  # 将相应的标签改为对应的marker
print(cm)

fig, ax = plt.subplots()

scatter = mscatter(x, y, c=c, m=cm, ax=ax, cmap=plt.cm.RdYlBu)

plt.show()