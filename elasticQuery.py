from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, F
import numpy as np

attributes = [
  "date",
  "serial_number",
  "model",
  "capacity_bytes",
  "failure"
]

smartAttributes = [
  {"statKey": "smart_1_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_5_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_7_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_9_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_9_raw", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_183_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_184_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_187_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_188_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_189_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_190_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_190_raw", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_193_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_193_raw", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_194_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_197_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
  {"statKey": "smart_198_normalized", "minValue": float("inf"), "maxValue": float("-inf")},
]

client = Elasticsearch('localhost:9201')

s = Search(using=client, index="backblaze_smart_data") \
    .query('query_string', query="(failure:1 OR date:2014-12-31) AND model:ST3000DM001") \
    .sort('date')

s = s[0:2776]

response = s.execute()

print("Response time: " + str(response.took) + "ms")

harddrives = []
harddriveStats = []

for hit in response.hits:
  harddrive = {
    "date": hit['date'],
    "serial_number": hit['serial_number'],
    "model": hit['model'],
    "capacity_bytes": hit['capacity_bytes'],
    "failure": hit['failure'],
    "smartStats": [float(hit['failure'])]
  }

  for stat in smartAttributes:
    driveValue = hit[stat['statKey']]
    if driveValue == "":
      driveValue = -12.0
    else:
      driveValue = float(driveValue)

      if driveValue < stat['minValue']:
        stat['minValue'] = driveValue
      elif driveValue > stat['maxValue']:
        stat['maxValue'] = driveValue

    harddrive['smartStats'].append(driveValue)

  harddrives.append(harddrive)
  harddriveStats.append(harddrive['smartStats'])

correlations = np.corrcoef(harddriveStats, None, 0)
np.set_printoptions(suppress=True)
print(correlations)

#print(harddrives)

# for stat in smartAttributes:
#   print("Stat: " + stat['statKey'])
#   print("Max Value: " + str(stat['maxValue']))
#   print("Min Value: " + str(stat['minValue']))
#   print("Range: " + str(stat['maxValue'] - stat['minValue']))
#   print()





















# import matplotlib.pyplot as plt
# plt.plot(days, models, 'ro')
# plt.show()

# from sklearn import cluster
# from mpl_toolkits.mplot3d import Axes3D


# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(data, target).predict(data)
# print("Number of mislabeled points out of a total %d points : %d for a %d percent failure" % (data.shape[0],(target != y_pred).sum(), ((target != y_pred).sum() / data.shape[0]) * 100))

# k = 2

# for k in range(2,10):
#   kmeans = cluster.KMeans(n_clusters=k)
#   kmeans.fit(data)

#   labels = kmeans.labels_
#   centroids = kmeans.cluster_centers_

#   from matplotlib import pyplot

#   for i in range(k):
#     #fig = pyplot.figure(1, figsize=(4, 3))
#     #pyplot.clf()
#     #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#     ds = data[np.where(labels==i)]
#     pyplot.plot(ds[:,0],ds[:,1],ds[:,2],ds[:,3],ds[:,4], 'o')
#     #ax.scatter(ds[:, 0], ds[:, 1], ds[:, 2], c=labels.astype(np.float))
#     lines = pyplot.plot(centroids[i,0], centroids[i,1], 'kx')
#     pyplot.setp(lines, ms=15.0)
#     pyplot.setp(lines, mew=2.0)

#   pyplot.show()
