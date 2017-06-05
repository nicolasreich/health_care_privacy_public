import pandas as pd            
import numpy as np 

"""
This module contains the function to compute k-anonymity using hierarchical taxonomy trees, as defined in
Anonymization by Local Recoding in Data with Attribute Hierarchical Taxonomies, J. Li, R. C.-W. Wong, A. W.-C. Fu, and J. Pei
as well as some helper functions.
"""
class TaxonomyTree:
    """
    Defines a node in a taxonomy tree. Each node contains its value, the weights between each level,
    its parent, and a function that should take a value of the corresponding category and return a boolean indicating
    if this value corresponds to this node. 
    """
    def __init__(self, value, weights, fun=None):
        self.value = value
        self.weights = weights
        self.parent = None
        self.level = 0
        self.children = {}
        # dirty hack to find nodes faster
        self.__fast_find = []
        if fun is None:
            self.fun = lambda x: x == value
        else:
            self.fun = fun
        self.__fast_find.append((self.fun, self))
        self.__root = self
    def create_child(self, value, fun=None):
        """
        Adds a child to the node. If there is already a child with the same value, it will be replaced.
        
        :param value: the value of the node
        :param fun: the function to test if a value should match this node. If none is provided, the default is equality with the value.
        :returns: the created child, to allow chained calls.
        """
        child = TaxonomyTree(value, self.weights, fun)
        self.children[value] = child
        child.parent = self
        child.level = self.level + 1
        child.__root = self.__root
        self.__root.__fast_find.append((child.fun, child))
        return child
    def find_node(self, value):
        """
        Finds the node to which a certain value corresponds, i.e. the one for which node.fun(value) = True
        
        :param value: the value to find
        :returns: the node corresponding to the value, or None if there aren't any
        """
        for (fun, node) in self.__root.__fast_find:
            if fun(value):
                return node
        return None

class TaxonomyError(LookupError):
    pass

def find_closest_common_ancestor(node1, node2):
    """
    Finds the closest common ancestor between two TaxonomyTree nodes
    
    :param node1: one of the nodes
    :param node2: the other
    :returns: the closest common ancestor, or None if they have no common ancestor
    """
    if node1 == node2:
        return node1
    elif node1.level == node2.level:
        return find_closest_common_ancestor(node1.parent, node2.parent)
    elif node1.level < node2.level:
        return find_closest_common_ancestor(node1, node2.parent)
    else:
        return find_closest_common_ancestor(node1.parent, node2)
    
def to_list(x):
    """
    If x is not an iterable or is a string, transforms it in a list
    
    :param x: anything really
    :returns: [x] if x is either a string or not an iterable, x otherwise
    """
    import collections
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
            x = [x]
    return x

def WHD(q, p, h, weights):
    """
    Weighted Hierarchical Distance between two levels, as defined in the paper
    
    :param q: one of the levels (an int)
    :param p: the other level
    :param h: the total height of the tree (an int)
    :param weights: the weights between the different levels of the trees (a 2-dim array of numbers)
    :returns: the weighted hierarchical distance between the two levels
    """
    
    num = 0.0
    for j in range(q, p):
        num += weights[j]
    den = 0.0
    for j in range(1, h):
        den += weights[j]
    return 0 if num == 0 else num/den
    

def compute_class_distance(e_class_1, n1, e_class_2, n2, columns, taxonomies):
    """
    Computes the distance between the two provided equivalence classes and their closest common ancestors
    
    :param e_class_i: the value of the equivalence classes, m-tuples with each entry corresponding to a column in columns
    :param ni: the size of each equivalence class
    :param columns: the list of columns that are being anonymized
    :param taxonomies: a dict of TaxonomyTrees, with one tree per column
    :returns: the distance between the two classes as defined in the paper and a m-tuple containing the closest common ancestor to each pair of value in e_class_i
    :raises TaxonomyError: if the taxonomy isn't complete and a node can't be found for a value of e_class_i
    """
    e_class_1 = to_list(e_class_1)
    e_class_2 = to_list(e_class_2)
    # find the nodes in the taxonomy trees to which each value from the class corresponds
    nodes_1 = list(map(lambda x: taxonomies[x[1]].find_node(x[0]), zip(e_class_1, columns)))
    nodes_2 = list(map(lambda x: taxonomies[x[1]].find_node(x[0]), zip(e_class_2, columns)))
    missing_taxonomy = 'Incomplete taxonomy: no node found for value {} in column {}'
    try:
        none_index = nodes_1.index(None)
        raise TaxonomyError(missing_taxonomy.format(e_class_1[none_index], columns[none_index]))
    except ValueError:
        # None not in nodes_1
        pass
    try:
        none_index = nodes_2.index(None)
        raise TaxonomyError(missing_taxonomy.format(e_class_2[none_index], columns[none_index]))
    except ValueError:
        # None not in nodes_2
        pass
    closest_common_ancestors = list(map(lambda x: find_closest_common_ancestor(x[0], x[1]), zip(nodes_1, nodes_2)))
    # distortion between e_class_1 and its closest common ancestors with e_class_2
    # distortion is the sum of the weighted hierarchical distance for each attribute
    distortion1 = sum(map(lambda x: WHD(x[1].level, x[0].level, len(x[0].weights), x[0].weights), zip(nodes_1, closest_common_ancestors)))
    # same for e_class_2
    distortion2 = sum(map(lambda x: WHD(x[1].level, x[0].level, len(x[0].weights), x[0].weights), zip(nodes_2, closest_common_ancestors)))
    # finally the distance
    return n1*distortion1+n2*distortion2, closest_common_ancestors

def anonymize(data, columns, taxonomies, k):
    """
    k-anonymization of a dataset using hierarchical taxonomies
    
    :param data: the original dataset, which should be a pandas dataframe
    :param columns: the columns to anonymize, i.e. the quasi-identifier
    :param taxonomies: a dict of TaxonomyTrees, with entries {c: T} where c is a column in columns and T a TaxonomyTree
    :param k: the k in k-anonymity
    :returns: the anonymized dataset
    :raises TypeError: if the type of an element's ancestor doesn't work with the dataframe. For example if in the TaxonomyTree an int has a string has parent, when inserting into the dataframe an exception will be raised. 
    """
    import random
    import math
    import itertools
    import time
    new_data = data.copy()
    # groupby the quasi-identifier to get the equivalence classes
    # this yields a series with columns as index and a list of patient numbers as data
    equivalence_classes = data.groupby(columns).apply(lambda x: list(x.index))
    # get those smaller than k
    smaller_classes = equivalence_classes.select(lambda x: len(equivalence_classes.loc[x]) < k)
    time_ancestors = 0
    time_classes = 0
    # let's remember some of our computations
    class_distances = {}
    while(not smaller_classes.empty):
        # take a random equivalence class
        random_class_index = smaller_classes.index[random.randint(0, len(smaller_classes)-1)]
        
        
        # find the closest equivalence class
        best_distance = math.inf
        for e_index in equivalence_classes.drop(random_class_index).index:
            combined_index = tuple(sorted(map(lambda x: tuple(sorted(x)), zip(e_index, random_class_index))))
            if combined_index in class_distances:
                distance, closest_common_ancestor = class_distances[combined_index]
            else:
                distance, closest_common_ancestors = compute_class_distance(random_class_index, 
                                                                        len(equivalence_classes.loc[[random_class_index]]),
                                                                        e_index, 
                                                                        len(equivalence_classes.loc[[e_index]]), 
                                                                        columns, 
                                                                        taxonomies)
                class_distances[combined_index] = (distance, closest_common_ancestors)
            if distance < best_distance:
                best_distance = distance
                closest_index = e_index
                closest_class_common_ancestors = closest_common_ancestors
                
        # generalize the classes
        small_class = equivalence_classes.loc[[random_class_index]]
        closest_class = equivalence_classes.loc[[closest_index]]
        # if the size of the small class + the closest is >= 2*k, the small class should be generalized only
        # with a portion of the closest class
        if len(small_class) + len(closest_class) >= 2*k:
            stub = closest_class[:k-len(small_class)]
            trunk = closest_class[k-len(small_class):]
        else:
            stub = closest_class
            trunk = None
        # remove them from the list of equivalence classes
        equivalence_classes.drop(random_class_index, inplace=True)
        equivalence_classes.drop(closest_index, inplace=True)
        smaller_classes.drop(random_class_index, inplace=True)
        if closest_index in smaller_classes:
            smaller_classes.drop(closest_index, inplace=True)
        if trunk is not None:
            equivalence_classes = equivalence_classes.set_value(tuple(list(map(to_list, to_list(closest_index)))), to_list(trunk))
        
        small_class = to_list(small_class)
        closest_class = to_list(closest_class)
        
        levels = list(map(lambda x: x.value, closest_class_common_ancestors))
        # if there is already an equivalence classes with this value, add the data points to the class
        new_class = list(itertools.chain.from_iterable(small_class.values + closest_class.values))
        if len(levels) == 1 and levels[0] in equivalence_classes.index:
            equivalence_classes.loc[levels[0]] += new_class
        elif tuple(levels) in equivalence_classes.index:
            equivalence_classes.loc[tuple(levels)] += new_class
        else:
            equivalence_classes = equivalence_classes.set_value(tuple(levels), new_class)
        # if the new class is smaller than k, add it back to the list of small classes
        if len(new_class) < k:
            if len(levels) == 1 and levels[0] in smaller_classes.index:
                smaller_classes.loc[levels[0]] += new_class
            elif tuple(levels) in smaller_classes.index:
                smaller_classes.loc[tuple(levels)] += new_class
            else:
                smaller_classes = smaller_classes.set_value(tuple(levels), new_class)
        for patient in new_class:
            for i, col in enumerate(columns):
                try:
                    new_data = new_data.set_value(patient, col, closest_class_common_ancestors[i].value)
                except ValueError:
                    pass
                    raise TypeError("The type of the taxonomy doesn't work with type of the column ",col)
                except TypeError as e:
                    print(e)
                    return (new_data, patient, new_class, i, col, closest_common_ancestors)
    return new_data