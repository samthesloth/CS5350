from codecs import namereplace_errors
from operator import truediv
import numpy as np
import pandas as pd
import sys
from math import e
from Node import Node

# Read in the csv file and save it as a pandas db
def readCsvData(name):
  return pd.read_csv(name, header=None)

# Check if all of the data points have the same label
def sameLabels(data):
  a = data.to_numpy()
  return (a[0] == a).all()

# Find the ME of the given column
def getME(data, column):
  labels = data[data.keys()[-1]].unique()
  values = data[column].unique()
  final = 0
  for val in values:
      merror = []
      for lab in labels:
          count = len(data[column][data[column]==val][data[data.keys()[-1]]==lab])
          total = len(data[column][data[column]==val])
          fraction = count/total
          if(fraction > 0):
            merror.append(fraction)
      fullFrac = total/len(data)
      final += fullFrac*min(merror)
  return final

# Get entropy of column. Quick and easy for total entropy
def getTotalEntropy(column):
  vc = pd.Series(column).value_counts(normalize=True, sort=False)
  return (-(vc * np.log2(vc))).sum()

#Find best attribute to split by
def findBestAttribute(data):
  entropies = []

  totalE = getME(data, data.keys()[-1])
  # Go through each column and get the entropy
  for column in data.keys()[:-1]:
    entropies.append(totalE-getME(data,column))
  
  # Return the index of the attribute with the lowest entropy
  return data.keys()[:-1][np.argmin(entropies)]

# Counts the current depth of the tree in case we have reached max
def countDepth(node):
  if node is None: return 1
  count = 2
  while(node.parent is not None):
    node = node.parent
    count += 1
  return count

# Recursively builds a tree from some root node
def id3Recurse(data, parent, maxDepth, cDepth=0):
  depth = countDepth(parent)
  if parent is None:
    parent = Node(None, None, False, None)
  # If everything has same label or max depth has been reached,
  if(sameLabels(data) or depth >= maxDepth or cDepth >= maxDepth):

    # If parent is root, no decision
    if parent.parent is None:
      dec = None
    else: dec = data[parent.label][0]

    #return node with rando label because most common wouldn't work
    return Node(data.iloc[-1, -1], parent, True, data.iloc[-1, -1])

  # Otherwise, we need to split by an attribute
  else: 
    attIndex = findBestAttribute(data)
    values = np.unique(data[attIndex])

    # If parent is root, make root's attribute decision this new attribute
    if parent.parent is None:
      parent.label = attIndex
    
    # Decision is parent at label
    dec = data[parent.label][0]

    # Creates a root node to put into recursive call
    n = Node(attIndex, parent, False, dec)

    # Goes through every value that attribute can take
    for val in values:
      temp = data[data[attIndex] == val].reset_index(drop=True)

      #if S_v is empty
      if(len(temp) == 0):
        # Simply add child with most common label
        n.children.append(Node(temp.iloc[: , -1].mode(), n, True, val))

      # Get counts of labels of attribute
      v,count = np.unique(temp.iloc[: , -1],return_counts=True)

      #If only one label for attribute value
      if len(count)==1:
        # Add child with that label
        n.children.append(Node(temp.iloc[-1, -1],n,True, val))
      #Otherwise, call recursively to get children of this root
      else:
        n.children.append(id3Recurse(temp, n, maxDepth, cDepth=+1))
    return n

#Print tree by printing the label of parent and decision of each child
def printTree(root):
  if(root is not None):
    if root.leaf:
      print(root.label)
    for c in root.children:
      print("~~~~~~~~~~~~~~~~~~~~~~~~")
      print(root.label)
      print(c.decision)
      print("~~~~~~~~~~~~~~~~~~~~~~~~")
      printTree(c)

# Recursively predict the label to give the instance, given the root of the tree
def predict(root, instance):
  ind = root.label

  # If it's a leaf, must go with label
  if root.leaf == True:
    return root.label

  # Get value at instance's attribute index
  feature_value = instance[ind]

  # For all children in this root, look for decision that matches
  for child in root.children:

    # If not a decision, must be label
    if child.decision is None:
      return child.label

    # If matches, recursively go to that node
    if feature_value == child.label or feature_value == child.decision :
      return predict(child, instance)
  return None

# Evaluate the percent correct that we predict that new data
def evaluate(root, data):
  y = 0
  t = 0

  # Go through each row in test data
  for index, row in data.iterrows():
    # Get predicion
    result = predict(root, data.iloc[index])
    # If prediction matches, incrememnt correct
    if result == data.iloc[: , -1].iloc[index]:
      y += 1
    t += 1
  accuracy = y / t
  return accuracy

# Read data
data = readCsvData(sys.argv[1])
# Get decision tree
root = id3Recurse(data, None, int(sys.argv[2]))
# Print success rate
print("Train: " + str(1 - evaluate(root, readCsvData("car/train.csv"))))
print("Test: " + str(1 - evaluate(root, readCsvData("car/test.csv"))))
