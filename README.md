# Support-Vector-Machines-
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns

data = pd.read_csv ('/storage/emulated/0/svm.csv')

sns.lmplot('x1', 'x2',

           data = data, 

           hue = 'r', 

          palette = 'Set1', 

           fit_reg = False, 

           scatter_kws = {"s" :50})

plt.show() 

#importing svm

from sklearn import svm

#converting columns as matrices

#formula for the hyperplane

#g(x) =W⃗₀x₁ + W⃗₁x₂ + b

points = data[['x1','x2']].values

result = data['r'] 

clf = svm.SVC(kernel = 'linear') 

clf.fit(points,result)

print('Vector of weights (w) = ', clf.coef_[0])

print('b =', clf.intercept_[0])

print('Indices of support vectors =', clf.support_)

print('Support vectors = ', clf.support_vectors_) 

print('Number of support vectors for each class = ', clf.n_support_) 

print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_)) 
#plotting the hyperplane

%matplotlib inline 

w = clf.coef_[0] #weight of vector

slope = -w[0] / w[1]

b = clf.intercept_[0]

#coordinates of the hyperplane 

xx = np.linspace(0, 4)

yy = slope * xx - (b / w[1]) #y = mx + c

#plotting the margins

s = clf.support_vectors_[0] #first support vector 

yy_below = slope * xx + (s[1] - slope * s[0])

# last support vector

s1 = clf.support_vectors_[-1]

yy_above = slope * xx + (s1[1] - slope * s1[0])

#plotting the points 

sns.lmplot('x1','x2', data = data, hue='r', 

           palette = 'Set1', 

           fit_reg = False, 

           scatter_kws={'s' : 70})

#plotting the hyperlane 

plt.plot(xx, yy, linewidth = 3, color = 'red') 

#plotting the two margins 

plt.plot(xx, yy_below, 'k--') 

plt.plot(xx, yy_above, 'k--')

#making predictions

print(clf.predict([[3,3]])[0])

print(clf.predict([[1,5]])[0])

print(clf.predict([[3,1]])[0])

#using the kernel trick when data is not linearly separable

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D 

from sklearn.datasets import make_circles

#X is features and C is class labels

X, c = make_circles(n_samples = 500, noise = 0.09)

rgb = np.array(['r', 'g']) 

plt.scatter(X[:, 0], X[:, 1], color = rgb[c]) 

%matplotlib inline 

plt.show() 

#adding the third axis

fig = plt.figure(figsize =(18,15)) 

ax = fig.add_subplot(111, projection = '3d')

z = X[:, 0]**2 + X[:, 1]**2

ax.scatter(X[:, 0], X[:, 1], z, color=rgb[c]) 

plt.xlabel("x-axis") 

plt.ylabel("y-axis") 

# combining X and z into a single array 

features = np.concatenate((X,z.reshape(-1,1)), axis =1)

#importing svm

from sklearn import svm

clf = svm.SVC(kernel='linear')

clf.fit(features, c) # is is the class labels

x3 = lambda x,y: (-clf.intercept_[0] - clf.coef_[0][0]*x - clf.coef_[0][1]*y)/ clf.coef_[0][2] 

tmp = np.linspace(-1.5,1.5,100)

x,y = np.meshgrid(tmp,tmp)

ax.plot_surface(x, y, x3(x,y), cstride=1)

plt.show()

