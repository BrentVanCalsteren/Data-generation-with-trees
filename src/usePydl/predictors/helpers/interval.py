from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Dict
import numpy as np


Interval = Tuple[bool, float, float, bool] #left_closed, startpoint, endpoint, right_closed


def contains_in_dict(inter_dic: Dict[List[Interval]], value, features):
    if isinstance(features, int):
        inter = inter_dic[features]
        for inte in inter:
            if contains_value(inte, value):
                return True
    else:
        table = [0] * len(features)
        for i, feature in enumerate(features):
            inter = inter_dic[features]
            for inte in inter:
                if contains_value(inte, value):
                    table[i] = 1
        return np.sum(table) == len(table)



def contains_value(inter: Interval, value):
    if not valid_interval(inter): return False
    left_closed, start, end, right_closed = inter
    if value < start or value > end: return False
    if value == start and not left_closed: return False
    if value == end and not right_closed: return False
    return True

def return_interval_as_list(inter):
    return np.array([inter[1], inter[2]])

def valid_interval(inter: Interval):
    left_closed, startpoint, endpoint, right_closed = inter
    if np.isnan(startpoint) or np.isnan(endpoint): return False
    if startpoint > endpoint: return False
    if startpoint == endpoint: return left_closed and right_closed
    return True


def intersect_union_interval_lists(inter_list_1: Sequence[Interval], inter_list_2: Sequence[Interval]):
    constraints = [iv for iv in inter_list_2 if valid_interval(iv)]
    if not constraints:
        return []

    new_list: List[Interval] = []
    for current in inter_list_1:
        for constraint in constraints:
            inter = intersect(current, constraint)
            if inter is not None:
                new_list.append(inter)
    return merge_intervals(new_list)


def union_interval_lists(inter_list_1: Sequence[Interval], inter_list_2: Sequence[Interval]):
    return merge_intervals(list(inter_list_1) + list(inter_list_2))


def intersect(inter_1: Interval, inter_2: Interval):
    if not (valid_interval(inter_1) and valid_interval(inter_2)): return None

    if inter_1[1] > inter_2[1]:
        start = inter_1[1]
        left_closed = inter_1[0]
    elif inter_1[1] < inter_2[1]:
        start = inter_2[1]
        left_closed = inter_2[0]
    else:
        start = inter_1[1]
        left_closed = inter_1[0] and inter_2[0]

    if inter_1[2] < inter_2[2]:
        end = inter_1[2]
        right_closed = inter_1[3]
    elif inter_1[2] > inter_2[2]:
        end = inter_2[2]
        right_closed = inter_2[3]
    else:
        end = inter_1[2]
        right_closed = inter_1[3] and inter_2[3]

    intersect_interval = (left_closed, start, end, right_closed)
    return intersect_interval if valid_interval(intersect_interval) else None

def merge_intervals(intervals: Sequence[Interval]):
    #merge intervals
    valid = [iv for iv in intervals if valid_interval(iv)]
    if not valid: return []
    #sort left closed bound
    valid.sort(key=lambda iv: (iv[1], not iv[0], iv[2]))
    merged: List[Interval] = [valid[0]]

    for iv in valid[1:]:
        last = merged[-1]
        #gap.
        if last[2] < iv[1]:
            merged.append(iv)
            continue
        #touching but on both open sides.
        if last[2] == iv[1] and not (last[3] or iv[0]):
            merged.append(iv)
            continue
        #overlap
        if iv[2] > last[2]:
            new_end = iv[2]
            new_right_closed = iv[3]
        elif iv[2] < last[2]:
            new_end = last[2]
            new_right_closed = last[3]
        else:
            new_end = last[2]
            new_right_closed = last[3] or iv[3]

        new_left_closed = last[0]
        if last[1] == iv[1]:
            new_left_closed = last[0] or iv[0]

        merged[-1] = (new_left_closed,last[1],new_end,new_right_closed)
    return merged