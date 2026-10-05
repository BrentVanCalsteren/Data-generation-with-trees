import copy
from typing import Dict
import numpy as np

import CONFIG
import src.usePydl.predictors.helpers.interval as INTERVAL

# ----------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------

class Tree:

    def __init__(self, tree):
        self.tree: Dict = tree
        self.intervals = None


    def get_intervals_based_on_sample(self, sample):

        def traverse(node, intervals):
            if "value" in node:
                return intervals

            elif "split_info" in node:
                split_info = node['split_info']
                features = split_info['feats']
                left_interval = split_info['left_interval']
                right_interval = split_info['right_interval']
                if INTERVAL.contains_in_dict(left_interval, sample[features],features):
                    intervals.append(left_interval)
                    return traverse(node['left'],intervals)
                elif INTERVAL.contains_in_dict(right_interval, sample[features],features):
                    intervals.append(right_interval)
                    return traverse(node['right'], intervals)
                else:
                    print(f'Impossible split, returning ref_sample value as interval, {sample}:{left_interval}:{right_interval}')
                    return [{features : [(True,sample[features],sample[features],True)]}]

            else: print("NEED TO REMAP THE TREE FIRST BEFORE SAMPLE LOOKUP")

        return traverse(self.tree, [])


    def get_intervals_based_on_directions(self, directions):
        intervals = []

        def traverse(node, idx):
            if "value" in node:
                return

            elif "split_info" in node:
                split_info = node['split_info']
                left_interval = split_info['left_interval']
                right_interval = split_info['right_interval']
                if directions[idx] == "L":
                    intervals.append(left_interval)
                    traverse(node['left'],  idx+1)
                elif directions[idx] == "R":
                    intervals.append(right_interval)
                    traverse(node['right'],  idx+1)
                else:
                    print('Impossible split')

            else: print("NEED TO REMAP THE TREE FIRST BEFORE SAMPLE LOOKUP")

        traverse(self.tree, 0)
        return intervals

    def traverse_leafID_through_tree(self, leaf_id):
        directions = []

        def recurse(node):
            if "value" in node:
                if node["value"]["leaf_id"] == leaf_id: return True
                return False

            if "left" in node:
                directions.append('L')
                if recurse(node["left"]): return True
                directions.pop()

            if "right" in node:
                directions.append('R')
                if recurse(node["right"]): return True
                directions.pop()

            return False

        recurse(self.tree)
        return directions

    def miss_class_error(self, samples):
        if self.intervals is None:
            print('CREATE INTERVALS FIRST')
            return 10e7
        sample_ids = np.zeros(samples.shape[0]).astype(str)
        for key, val in self.intervals.items():
            for i,sample in enumerate(samples):
                if INTERVAL.contains_in_dict(val,sample): sample_ids[i] = key
        _, counts = np.unique(sample_ids, return_counts=True)
        cs = np.sum(counts)
        return (cs - np.max(counts))/cs

    def get_depth(self):
        def recurse(node, depth):
            if "value" in node:return depth

            left_depth = depth
            right_depth = depth

            if "left" in node: left_depth = recurse(node["left"], depth + 1)

            if "right" in node: right_depth = recurse(node["right"], depth + 1)

            return max(left_depth, right_depth)

        return recurse(self.tree, 0)

    def get_leafs(self):
        leaves = []

        def recurse(node):
            if "value" in node:
                leaves.append(node)
            else:
                if "left" in node: recurse(node["left"])
                if "right" in node: recurse(node["right"])

        recurse(self.tree)
        return leaves



    def create_intervals_each_paths(self):
        intervals_each_path = {}

        def path_finder(node, current_intervals):
            if "value" in node:
                merged_intervals = {}
                for interval in current_intervals:
                    for key, val in interval.items():
                        if key in merged_intervals:
                            merged_intervals[key] = INTERVAL.intersect_union_interval_lists(merged_intervals[key], val)
                        else:
                            merged_intervals[key] = val
                intervals_each_path[node['value']['leaf_id']] = merged_intervals
                return


            elif "split_info" in node:
                split_info = node['split_info']
                left_interval = split_info['left_interval']
                right_interval = split_info['right_interval']
                if "left" in node:
                    current_intervals.append(left_interval)
                    path_finder(node["left"], current_intervals)
                    current_intervals.pop()
                if "right" in node:
                    current_intervals.append(right_interval)
                    path_finder(node["right"], current_intervals)
                    current_intervals.pop()
            else: print("NEED TO REMAP THE TREE FIRST BEFORE SAMPLE LOOKUP")

        path_finder(self.tree,[])
        self.intervals = intervals_each_path
        return intervals_each_path

#GLOBAL FUNCTIONS

def remap_tree(tree, feature_history):

    def recurse(node):
        if "value" in node: return

        node["split_info"] = feature_history.store_split_data(node["feat"])

        if "left" in node: recurse(node["left"])
        if "right" in node: recurse(node["right"])

    recurse(tree)
    return tree

def extend_tree(tree, hist_tree, leaf_id):
    def merge(node):
        if "value" in node:
            l_id = node["value"]["leaf_id"]
            if leaf_id == l_id:
                print("MERGED!")
                return hist_tree
            return node

        if "left" in node: node["left"] = merge(node["left"])
        if "right" in node: node["right"] = merge(node["right"])

        return node

    return merge(tree)




def score_path(path, intervals_each_feature, path_error, rel_prob):

    def count_splits(path_dict) -> int:
        if not path_dict: return 0
        total = 0
        for value in path_dict.vals():
            try:
                total += len(value[0])
            except Exception:
                pass
        return int(total)

    n_splits = float(len(path['directions']))
    total_len = 0.0
    for feat_id in range(n_features):
        total_len += intervals_each_feature[feat_id].get_total_lenght()

    score = (n_splits + 1.0) * float(rel_prob)
    score /= ((1.0 + float(path_error)) * (1.0 + total_len))
    return float(score)
