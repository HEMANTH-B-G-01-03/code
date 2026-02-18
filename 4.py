import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data= load_iris()
x=data.data #featurematrix
y=data.target
#split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
id3_tree=DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_tree.fit(x_train, y_train)
y_pred=id3_tree.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print(f'Accuracy of the ID# Decision tree:{accuracy*100: .2f}%')
print('\nDecision Tree Rules:')
print(export_text(id3_tree,feature_names=data.feature_names))
plt.figure(figsize=(12,8))
plot_tree(id3_tree, feature_names=data.feature_names, class_names=list(data.target_names),
filled=True)
plt.title("ID3 Descision Tree")
plt.show() 