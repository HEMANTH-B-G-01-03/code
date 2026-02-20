
import numpy as np
import matplotlib.pyplot as plt
def locally_weighted_regression(x,y, tau, x_query):
 m= len(x)
 w=np.exp( -np.sum((x-x_query)**2, axis=1)/(2*tau**2))
 w=np.diag(w)
 x_b= np.c_[np.ones((m,1)),x]
 theta= np.linalg.pinv(x_b.T @ w @ x_b) @x_b.T @ w @y
 x_query_b= np.array([1, x_query])
 return x_query_b @ theta
np.random.seed(0)
x=2*np.random.rand(100,1)
y=(3+ 2*x +np.random.randn(100,1)).flatten()
x=x.flatten()
sort_idx= np.argsort(x)
x=x[sort_idx]
y=y[sort_idx]
tau=0.1
x_test= np.linspace(x.min(), x.max(), 100)
#perform predictions
y_pred =[locally_weighted_regression(x[:, np.newaxis], y, tau, x_query) for x_query in
x_test]
#visualization
plt.figure(figsize=(10,6))
plt.scatter(x,y,color='blue',label='Data points')
plt.plot(x_test, y_pred, color='red', label='LWR fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show() 




















import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge

# generate data
np.random.seed(0)
x = 2 * np.random.rand(100)
y = 3 + 2 * x + np.random.randn(100)

# sort for smooth curve
idx = np.argsort(x)
x, y = x[idx], y[idx]

# reshape for sklearn
x = x.reshape(-1, 1)

# Locally weighted regression using RBF kernel
model = KernelRidge(kernel='rbf', gamma=1/(2*0.1**2))
model.fit(x, y)

# predictions
x_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_pred = model.predict(x_test)

# plot
plt.scatter(x, y, label="Data points")
plt.plot(x_test, y_pred, color="red", label="LWR fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()