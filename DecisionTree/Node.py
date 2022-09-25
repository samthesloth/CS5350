from inspect import Attribute


class Node:
    def __init__(self, label, parent, children, decision):
      self.label = label
      if len(children) == 0: self.leaf = True
      else: self.leaf = False
      self.children = children
      self.parent = parent
      self.decision = decision

    def __init__(self, label, parent, leaf, decision):
      self.label = label
      self.leaf = leaf
      self.children = []
      self.parent = parent
      self.decision = decision