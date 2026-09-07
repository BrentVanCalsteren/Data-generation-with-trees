from typing import Dict
import numpy as np

from splits import splits


# ----------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------

class Tree:

    def __init__(self, tree):
        self.tree: Dict = tree


    def let_sample_treverse_tree(self, sample):
        path = {'splits' : [],
                'directions': [],
                'leaf_vals': None}

        def traverse(node):
            if "value" in node:
                path['leaf_vals'] = node['value']
                return

            elif "split_info" in node:
                split_info = node['split_info']
                idx = split_info['id']
                split_obj = split_info['split_obj']
                direction = split_obj.evaluate_sample(idx,sample)
                path['splits'].append(split_info)
                path['directions'].append(direction)
                if direction == 'L':
                    traverse(node['left'])
                else:
                    traverse(node['right'])

            else: print("can't acces split_obj, remap first")

        traverse(self.tree)
        return path

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


    def get_all_paths(self):
        all_paths = []

        def copy_path(p):
            return {k: v.copy() for k, v in p.items()}

        def path_finder(node, current_path):
            if "value" in node:
                current_path['leaf_vals'] = node['value']
                all_paths.append(copy_path(current_path))
                return


            if "left" in node:
                split_info = node['split_info']
                current_path['splits'].append(split_info)
                current_path['directions'].append('L')
                path_finder(node["left"], current_path)
                current_path['splits'].pop()
                current_path['directions'].pop()

            if "right" in node:
                split_info = node['split_info']
                current_path['splits'].append(split_info)
                current_path['directions'].append('R')
                path_finder(node["right"], current_path)
                current_path['splits'].pop()
                current_path['directions'].pop()

        path_finder(self.tree,
                {'splits' : [],
                'directions': [],
                'leaf_vals': None})

        if len(all_paths) != len(self.get_leafs()):
            print("LEN PATHS AND LEAVES SHOULD BE THE SAME")

        return all_paths

#GLOBAL FUNCTIONS

def remap_tree(tree, feature_history):

    def recurse(node):
        if "value" in node: return


        split_data = feature_history.store_split_data(node["feat"])
        node["split_info"] = split_data

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
