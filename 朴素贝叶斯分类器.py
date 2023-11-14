import numpy as np
from sklearn.naive_bayes import CategoricalNB

x_train = np.random.randint(5, size=(10,10))
y_train = np.random.randint(1, high=11, size=(10,))
x_test = np.random.randint(5, size=(1,10))

#---------------------调包实现-------------------------
clf = CategoricalNB()
clf.fit(x_train, y_train)
print("调包实现的分类结果为：",clf.predict(x_test))

#---------------------自己实现-------------------------
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0)
            }
    
    def _calculate_likelihood(self, mean, var, x):
        eps = 1e-9  # 很小的平滑值，避免除零错误
        return np.exp(-((x - mean) ** 2) / (2 * (var + eps))) / np.sqrt(2 * np.pi * (var + eps))
    
    def _calculate_prior(self, c, X):
        likelihood = self._calculate_likelihood(self.parameters[c]['mean'], self.parameters[c]['var'], X)
        return np.sum(np.log(likelihood + 1e-9))  # 避免数值下溢
    
    def predict(self, X):
        y_pred = [self._calculate_prior(c, X) for c in self.classes]
        return self.classes[np.argmax(y_pred)]

# 模型训练
model = GaussianNaiveBayes()
model.fit(x_train, y_train)
predicted_class = model.predict(x_test)
print("自己实现的分类结果为：",predicted_class)



