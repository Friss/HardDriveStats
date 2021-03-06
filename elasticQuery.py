from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, F
import numpy as np
from sklearn.cross_validation import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from sklearn import tree
from sklearn import cluster
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA

attributes = [
  "date",
  "serial_number",
  "model",
  "capacity_bytes",
  "failure"
]

smartAttributes = [
  {"statKey": "smart_187_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_193_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
]

client = Elasticsearch('the.student.rit.edu:9201')

s = Search(using=client, index="backblaze_smart_data") \
    .query('query_string', query="(failure:1 OR date:2014-12-31) AND model:ST3000DM001") \
    .sort('date')

s = s[0:2776]

response = s.execute()

print("Response time: " + str(response.took) + "ms")

harddrives = []
harddriveStats = []
target = []
smart1 = []
smart1Failed = []
smart1Good = []
smart2 = []
smart2Failed = []
smart2Good = []

for hit in response.hits:
  harddrive = {
    "date": hit['date'],
    "serial_number": hit['serial_number'],
    "model": hit['model'],
    "capacity_bytes": hit['capacity_bytes'],
    "failure": hit['failure'],
    "smartStats": []
  }

  for stat in smartAttributes:
    driveValue = hit[stat['statKey']]
    if driveValue == "":
      driveValue = -1.0
    else:
      driveValue = float(driveValue)

      if driveValue < stat['minValue']:
        stat['minValue'] = driveValue
      elif driveValue > stat['maxValue']:
        stat['maxValue'] = driveValue

    harddrive['smartStats'].append(driveValue)

    if stat['statKey'] == 'smart_187_normalized':
      smart1.append(driveValue)
      if float(hit['failure']) == 1.0:
        smart1Failed.append(driveValue)
      else:
        smart1Good.append(driveValue)
    elif stat['statKey'] == 'smart_193_normalized':
      smart2.append(driveValue)
      if float(hit['failure']) == 1.0:
        smart2Failed.append(driveValue)
      else:
        smart2Good.append(driveValue)

  harddrives.append(harddrive)
  harddriveStats.append(harddrive['smartStats'])
  target.append(float(hit['failure']))



# correlations = np.corrcoef(harddriveStats, None, 0)
# np.set_printoptions(suppress=True)
# print(correlations)

#print(harddrives)

# for stat in smartAttributes:
#   print("Stat: " + stat['statKey'])
#   print("Max Value: " + str(stat['maxValue']))
#   print("Min Value: " + str(stat['minValue']))
#   print("Range: " + str(stat['maxValue'] - stat['minValue']))
#   print()


harddriveStats = np.array(harddriveStats)
target = np.array(target)


###
#
# Setup 10 Cross Fold Validation
#
###
X = harddriveStats
y = target

kFold = KFold(len(target), n_folds = 10, shuffle=True)


###
#
# Naive Bayes Classifier
#
###
# gnb = GaussianNB()
# bestScore = float("-inf")
# for train_index, test_index in kFold:
#   score = gnb.fit(X[train_index], y[train_index]).score(X[test_index], y[test_index])
#   print(score)
#   print()
#   if score > bestScore:
#     bestScore = score

# print("Best Score: "+ str(bestScore))

###
#
# Linear Regression Classifier
#
###
# gnb = LinearRegression()
# bestScore = float("-inf")
# for train_index, test_index in kFold:
#   score = gnb.fit(X[train_index], y[train_index]).score(X[test_index], y[test_index])
#   print(score)
#   print()
#   if score > bestScore:
#     bestScore = score

# print("Best Score: "+ str(bestScore))

###
#
# Decision Tree Classifier
#
###
# best = float("-inf")
# bestCrit = ""
# bestIndex = -1
# index = 0
# for train_index, test_index in kFold:
#   for criterion in ["gini", "entropy"]:
#     clf = tree.DecisionTreeClassifier(criterion=criterion)
#     clf = clf.fit(X[train_index], y[train_index])
#     score = clf.score(X[test_index], y[test_index])
#     print(criterion + ": " + str(score))
#     print()
#     if score > best:
#       best = score
#       bestCrit = criterion
#       bestIndex = index
#     from sklearn.externals.six import StringIO
#     with open("harddrive"+str(index)+criterion+".dot", "w") as f:
#       f = tree.export_graphviz(clf, out_file=f)
#   index += 1

# print("Best ("+bestCrit+") "+str(bestIndex)+" : " + str(best))

# clf = clf.fit(harddriveStats, target)
# from sklearn.externals.six import StringIO
# with open("harddriveFull.dot", "w") as f:
#   f = tree.export_graphviz(clf, out_file=f)



###
#
# Graphs of data
# All data
# Failed Drives
# Good Drives
#
###
# pyplot.plot(smart1, smart2, 'ro')
# pyplot.title("S.M.A.R.T. 187 vs S.M.A.R.T. 193 for ST3000DM001")
# pyplot.xlabel("S.M.A.R.T. 187 Normalized")
# pyplot.ylabel("S.M.A.R.T. 193 Normalized")
# pyplot.grid(True)
# pyplot.axis([-5, 105, -5, 105])
# pyplot.show()

# pyplot.plot(smart1Failed, smart2Failed, 'go')
# pyplot.title("S.M.A.R.T. 187 vs S.M.A.R.T. 193 for ST3000DM001")
# pyplot.xlabel("S.M.A.R.T. 187 Normalized")
# pyplot.ylabel("S.M.A.R.T. 193 Normalized")
# pyplot.grid(True)
# pyplot.axis([-5, 105, -5, 105])
# pyplot.show()


# pyplot.plot(smart1Good, smart2Good, 'bo')
# pyplot.title("S.M.A.R.T. 187 vs S.M.A.R.T. 193 for ST3000DM001")
# pyplot.xlabel("S.M.A.R.T. 187 Normalized")
# pyplot.ylabel("S.M.A.R.T. 193 Normalized")
# pyplot.grid(True)
# pyplot.axis([-5, 105, -5, 105])
# pyplot.show()


###
#
# KMeans Clustering
#
###
# for k in range(2,10):
#   kmeans = cluster.KMeans(n_clusters=k)
#   kmeans.fit(harddriveStats)

#   labels = kmeans.labels_
#   centroids = kmeans.cluster_centers_



#   for i in range(k):
#     ds = harddriveStats[np.where(labels==i)]
#     pyplot.plot(ds[:,0],ds[:,1], 'o')

#     lines = pyplot.plot(centroids[i,0], centroids[i,1], 'kx')
#     pyplot.setp(lines, ms=15.0)
#     pyplot.setp(lines, mew=2.0)

#   pyplot.title(str(k) + " Means Clustering of S.M.A.R.T. 187 vs S.M.A.R.T. 193 for ST3000DM001")
#   pyplot.xlabel("S.M.A.R.T. 187 Normalized")
#   pyplot.ylabel("S.M.A.R.T. 193 Normalized")
#   pyplot.grid(True)
#   pyplot.axis([-5, 105, -5, 105])

#   pyplot.show()


###
#
# K Nearest Neighbors Classifier
#
###
# best = float("-inf")
# bestN = 0
# bestWeight = ""

# worst = float("inf")
# worstN = 0
# worstWeight = ""

# for train_index, test_index in kFold:
#   n_neighborsX = []
#   accUni = []
#   accDist = []
#   for n_neighbors in range(1, 25):
#     n_neighborsX.append(n_neighbors)
#     for weights in ['uniform', 'distance']:
#         # we create an instance of Neighbours Classifier and fit the data.
#         clf = KNeighborsClassifier(n_neighbors, weights=weights)
#         clf.fit(X[train_index], y[train_index])
#         score = clf.score(X[test_index], y[test_index])
#         if score > best:
#           best = score
#           bestN = n_neighbors
#           bestWeight = weights
#         elif score < worst:
#           worst = score
#           worstN = n_neighbors
#           worstWeight = weights

#         print(str(n_neighbors) + " : (" + weights + ") " + str(score))


#         if weights == "uniform":
#           accUni.append(score)
#         else:
#           accDist.append(score)

#     print()

#   pyplot.plot(n_neighborsX, accUni, 'ro')
#   pyplot.plot(n_neighborsX, accDist, 'bo')
#   pyplot.title("Accuracy vs K Nearest Neighbors (Uniform and Distance Weights)")
#   pyplot.xlabel("K Values")
#   pyplot.ylabel("Accuracy")
#   pyplot.grid(True)
#   pyplot.show()



# print(str(bestN) + " " + str(best) + " " + str(bestWeight))
# print(str(worstN) + " " + str(worst) + " " + str(worstWeight))


# h = .02  # step size in the mesh

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
#          "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     LDA(),
#     QDA()]
# figure = plt.figure(figsize=(54, 6))
# i = 1
# # preprocess dataset, split into training and test part
# X = harddriveStats
# y = target
# X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# # just plot the dataset first
# cm = plt.cm.RdBu
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# ax = plt.subplot(1, len(classifiers) + 1, i)
# # Plot the training points
# ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# # and testing points
# ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xticks(())
# ax.set_yticks(())
# i += 1

# # iterate over classifiers
# for name, clf in zip(names, classifiers):
#     ax = plt.subplot(1, len(classifiers) + 1, i)
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)

#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, m_max]x[y_min, y_max].
#     if hasattr(clf, "decision_function"):
#         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     else:
#         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

#     # Plot also the training points
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
#     # and testing points
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
#                alpha=0.6)

#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(name)
#     ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#             size=15, horizontalalignment='right')
#     i += 1

# figure.subplots_adjust(left=.02, right=.98)
# plt.show()
