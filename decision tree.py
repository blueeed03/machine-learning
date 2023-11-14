import numpy as np
from sklearn import tree
import graphviz

x = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = [0,1,1,1,2,3,3,4]

#训练决策树模型
clf = tree.DecisionTreeClassifier()
clf.fit(x, y)

#测试集
print(clf.predict([[1,0,0]]))

#生成决策树示意图的PDF文件
dot_deta = tree.export_graphviz(clf, out_file = None)
graph = graphviz.Source(dot_deta)
graph.render('result')


