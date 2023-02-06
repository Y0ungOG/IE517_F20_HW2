import numpy as np
import pandas as pd
data = pd.read_csv('/Users/maochenyi/Desktop/Treasury Squeeze raw score data.csv')
data.head()

#give class to dataset,change from t/f to 0/1
data['class'] = data['squeeze'].replace({True:1,False:0})
y =  data['class']
x = data.iloc[:,[3,4]]

#split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 6, stratify = y)

#the formula of indicators
import matplotlib.pyplot as plt
def gini(p):
    return p*(1-p) +(1-p)*(1-(1-p))

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def error(p):
    return 1-np.max([p,1-p])

x = np.arange(0.0, 1.01, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()

ax = plt.subplot(111)

for i, lab , ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy','Entropy(scaled)','Gini','Miscla Error'],
                           ['-','-','--','-.'],
                           ['black','lightgray','red','green','cyan']):
    line = ax.plot(x, i, label = lab, linestyle = ls, lw = 2, color = c)

ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15),
          ncol = 5, fancybox = True, shadow = False)
ax.axhline(y = 0.5, linewidth = 1, color = 'k', linestyle = '--')
ax.axhline(y = 1, linewidth = 1, color = 'k', linestyle = '--') 
plt.ylim([0,1.1])
plt.xlabel('p(i= 1)')
plt.ylabel('Impurity')
plt.show()


#Building Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion= 'gini',
                              max_depth=(4),
                              random_state = 6)
tree.fit(x_train,y_train)
x_combined  = np.vstack((x_train,x_test))
y_combined = np.hstack((y_train,y_test))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x_combined, 
                      y_combined, 
                      clf = tree)
plot_decision_regions()

#Create the decision tree image
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree, filled = True, class_names= ['squeeze','not squeeze'],feature_names=['price distortion','roll start'],out_file = None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators = 25, random_state= 6, n_jobs = 2)
forest.fit(x_train,y_train)
plot_decision_regions(x_combined, y_combined, clf = forest)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1,600)
scores =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

best_score_index = max(enumerate(scores), key=lambda x: x[1])[0]
best_k = k_range[best_score_index]

knn = KNeighborsClassifier(n_neighbors=265,p=2)

#Standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

knn.fit(x_train_std,y_train)
plot_decision_regions(x_combined, y_combined, clf = knn)

print("My name is {Chenyi Mao}")
print("My NetID is: {chenyim2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



