"""
This module contains the function to compute k-anonymity using mondrian multi-dimensionality, as defined in
Mondrian Multidimensional K-Anonymity, Kristen LeFevre, David J. DeWitt, Raghu Ramakrishnan
as well as some helper functions.
"""

def cut_partition(partition, columns, k, ranges):
    """
    Cuts a partition into two smaller ones if there is an allowable cut
    
    :param partition: the partition to cut, a pandas dataframe
    :param columns: the columns to anonymize, i.e. the quasi-identifier
    :param k: the k in k-anonymity
    :param ranges: the range of each column before any partitioning, i.e. column.max - column.min
    :returns: the two smaller partition if there was an allowable cut, or None otherwise
    """
    def normalized_range(df, column, ranges):
        return 0 if ranges[column] == 0 else (df[column].max()-df[column].min())/ranges[column]
        #print(df[column].max(), df[column].min(), ranges[column])
        #return n_range
    
    sorted_columns = sorted(columns, key=lambda x: normalized_range(partition, x, ranges), reverse=True)
    for dim in sorted_columns:
        median = partition[dim].median()
        lhs = partition[partition[dim] <= median]
        rhs = partition[partition[dim] > median]
        if len(lhs.index) >= k and len(rhs.index) >= k:
            return (lhs, rhs)
    return None

def recode(partition, columns):
    """
    Recodes a partition, replacing every value by a summary statistic
    
    :param partition: the partition to recode
    :param columns: the columns to anonymize
    :returns: a new partition with the modified values
    """
    part = partition.copy()
    for dim in columns:
        minimum = part[dim].min()
        maximum = part[dim].max()
        if(minimum == maximum): 
            summary = str(minimum)
        else:
            summary = str(minimum) + '-' + str(maximum)
        part[dim] = part[dim].map(lambda x: summary)
    return part


def anonymize(partition, columns, k):
    """
    A top-down greedy algorithm for strict multidimensional partitioning
    
    :param partition: the dataset to anonymize, a pandas datafram
    :param columns: the columns to anonymize, i.e. the quasi-identifier
    :param k: the k in k-anonymity
    :returns: the anonymized dataframe
    """
    def anonymize_helper(partition, columns, k, ranges):
        """
        Internal helper function with the ranges for each colums
        """
        partitions = cut_partition(partition, columns, k, ranges)
        if partitions is None:
            return recode(partition, columns)
        else:
            (lhs, rhs) = partitions
            return anonymize_helper(lhs, columns, k, ranges).append(anonymize_helper(rhs, columns, k, ranges))
    
    
    ranges = {}
    for dim in columns:
        ranges[dim] = partition[dim].max()-partition[dim].min()
    return anonymize_helper(partition, columns, k, ranges)