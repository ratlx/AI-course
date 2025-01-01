import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn import tree

# Load the data
data = pd.read_csv('ex3data.csv')

# Preprocessing the categorical columns
data['天气'] = data['天气'].map({'好': 1, '坏': 0})
data['是否周末'] = data['是否周末'].map({'是': 1, '否': 0})
data['是否有促销'] = data['是否有促销'].map({'是': 1, '否': 0})
data['销量'] = data['销量'].map({'高': 1, '低': 0})

# Features and target variable
X = data[['天气', '是否周末', '是否有促销']]
y = data['销量']

# Train a decision tree classifier (CART by default)
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X, y)

# Print the tree
print(export_text(clf, feature_names=['天气', '是否周末', '是否有促销']))

# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=['weather', 'isWeekend', 'hasSale'], class_names=['low', 'high'], filled=True)
plt.show()