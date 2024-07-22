#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
for point in data:
  salary = point[0]
  bonus = point[1]
  matplotlib.pyplot.scatter( salary, bonus )
  
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


### your code below
target, inputs = targetFeatureSplit(data)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(inputs, target)

predictions = reg.predict(inputs)
cleaned_data = []
for i in range(len(predictions)):
  error = abs(target[i] - predictions[i])
  cleaned_data.append((inputs[i], target[i], error))
    
print(sorted(cleaned_data, key=lambda tup: tup[2])[len(cleaned_data)-2:], '\n')

# import matplotlib.pyplot as plt
# plt.plot(inputs, reg.predict(inputs), color="blue")

# plt.scatter( inputs, target, color='b' )

# plt.xlabel(features[1])
# plt.ylabel(features[0])
# # plt.legend()
# plt.show()
# # plt.plot( features, reg.predict(feature_test) )