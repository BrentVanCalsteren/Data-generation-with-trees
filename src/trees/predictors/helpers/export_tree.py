from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ==============================================================================
# 5. Primary All-in-One Function
# ==============================================================================

def export_tree_analysis(
    trees: Union[List[Dict],Dict], n_features: int, #tree is result from .get_complete_tree when called in feature history
    folder_path =  Path("tree_data"),
    output_image_path = Path("trees_visualization.png"),
    output_csv_path = Path("ntree_x_nfeatures.csv"),
    title = "image trees", dpi: int = 300):

    if trees is isinstance(trees, Dict):
        trees = [trees]

    feature_counts = np.zeros((len(trees), n_features),dtype=np.float32)
    for i, tree in enumerate(trees):
        split_objs = get_all_split_obj(tree)
        feature_counts[i] = get_feat_counts(split_objs)

    export_feature_matrix(matrix=feature_counts, csv_path=folder_path / output_csv_path)

    saved_img_path = visualize_all_trees(trees=trees, image_path=output_image_path,
        title=title, dpi=dpi)


def get_all_split_obj(tree):
    pass

def get_feat_counts(split_obj):
    pass




def visualize_all_trees(
    trees: Any,
    image_path: Union[str, Path] = "trees_visualization.png",
    feature_names: Optional[List[str]] = None,
    use_original_features: bool = True,
    title: Optional[str] = "Decision Trees Visualization",
    max_cols: int = 3,
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None
):

    tree_dicts = normalize_trees_input(trees)
    n_trees = len(tree_dicts)

    # Grid layout calculation
    n_cols = min(max_cols, max(1, n_trees))
    n_rows = math.ceil(n_trees / n_cols)

    if figsize is None:
        sub_width = 6.0
        sub_height = 4.8
        figsize = (sub_width * n_cols, sub_height * n_rows + (0.8 if title else 0.2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor("#FAFAFA")

    # Render each tree on its dedicated subplot
    for idx in range(n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        if idx < n_trees:
            sub_title = f"Tree {idx + 1} of {n_trees}" if n_trees > 1 else "Decision Tree"
            draw_single_tree(
                tree_dicts[idx],
                ax=ax,
                title=sub_title,
                feature_names=feature_names,
                use_original_features=use_original_features
            )
        else:
            ax.axis("off")  # hide unused subplots

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.93)

    out_image = Path(image_path)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_image, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[tree_exporter] Saved image displaying all {n_trees} tree(s) to: '{out_image}'")
    return out_image


def extract_node_feature_id(node: Dict[str, Any], use_original_features: bool = True) -> Optional[int]:
    if not isinstance(node, dict):
        return None

    # Check for remapped split_obj first
    if use_original_features and "split_obj" in node:
        split_obj = node["split_obj"]
        if hasattr(split_obj, "feats") and split_obj.feats is not None:
            feats = split_obj.feats
            if hasattr(feats, "item"):
                return int(feats.item())
            elif isinstance(feats, (list, tuple, np.ndarray)) and len(feats) > 0:
                first = feats[0]
                return int(first.item()) if hasattr(first, "item") else int(first)
            return int(feats)

    # Fallback to direct 'feat' attribute
    if "feat" in node and node["feat"] is not None:
        feat = node["feat"]
        if hasattr(feat, "item"):
            return int(feat.item())
        return int(feat)

    return None


def is_leaf_node(node: Dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    has_left = "left" in node and node["left"] is not None
    has_right = "right" in node and node["right"] is not None
    return ("value" in node or "error" in node) and (not has_left and not has_right)


def count_feature_splits_single_tree(
    tree_dict: Dict[str, Any],
    use_original_features: bool = True
) -> Dict[int, int]:

    counts: Dict[int, int] = {}

    def traverse(node: Any):
        if not isinstance(node, dict):
            return
        if is_leaf_node(node):
            return

        feat_id = extract_node_feature_id(node, use_original_features=use_original_features)
        if feat_id is not None:
            counts[feat_id] = counts.get(feat_id, 0) + 1

        if "left" in node:
            traverse(node["left"])
        if "right" in node:
            traverse(node["right"])

    traverse(tree_dict)
    return counts


def get_ntree_x_nfeatures_matrix(
    trees: Any,
    n_features: Optional[int] = None,
    feature_names: Optional[List[str]] = None,
    use_original_features: bool = True,
) -> pd.DataFrame:
    tree_dicts = normalize_trees_input(trees)
    n_trees = len(tree_dicts)

    # Gather split counts for every tree
    all_tree_counts: List[Dict[int, int]] = [
        count_feature_splits_single_tree(t, use_original_features=use_original_features)
        for t in tree_dicts
    ]

    # Determine number of features
    if feature_names is not None:
        total_feats = len(feature_names)
    elif n_features is not None:
        total_feats = int(n_features)
    else:
        # Infer max feature ID from encountered trees
        max_feat = -1
        for counts in all_tree_counts:
            if counts:
                max_feat = max(max_feat, max(counts.keys()))
        total_feats = max(max_feat + 1, 1)

    # Build dense matrix
    matrix = np.zeros((n_trees, total_feats), dtype=int)
    for tree_idx, counts in enumerate(all_tree_counts):
        for feat_id, cnt in counts.items():
            if 0 <= feat_id < total_feats:
                matrix[tree_idx, feat_id] = cnt

    # Column labels
    if feature_names is not None and len(feature_names) == total_feats:
        columns = list(feature_names)
    else:
        columns = [f"Feature_{j}" for j in range(total_feats)]

    # Row labels
    index = [f"Tree_{i}" for i in range(n_trees)]

    return pd.DataFrame(data=matrix, index=index, columns=columns)


# ==============================================================================
# 3. Export Utilities (CSV & Excel)
# ==============================================================================

def export_feature_matrix(matrix: np.ndarray, csv_path: Path):

    results: Dict[str, Optional[Path]] = {"csv": None, "excel": None}
    # CSV Export
    if csv_path:
        out_csv = Path(csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=True)
        results["csv"] = out_csv
        print(f"[tree_exporter] Exported ntree_x_nfeatures matrix to CSV: '{out_csv}' (shape: {df.shape})")
    return results


# ==============================================================================
# 4. Tree Layout & Matplotlib Visualization
# ==============================================================================

def _compute_hierarchical_layout(node: Dict[str, Any], depth: int = 0, next_x: Optional[List[float]] = None):
    """
    Computes (x, y) coordinates for each node in a binary tree.
    Leaves receive monotonic horizontal coordinates, and parent nodes are centered
    above their respective children.
    """
    if next_x is None:
        next_x = [0.0]

    positions = {}
    node_key = id(node)

    if is_leaf_node(node):
        x = next_x[0]
        next_x[0] += 1.0
        positions[node_key] = (x, -depth, node, True)
        return positions, x, depth

    has_left = "left" in node and isinstance(node["left"], dict)
    has_right = "right" in node and isinstance(node["right"], dict)

    max_d = depth
    left_x: Optional[float] = None
    right_x: Optional[float] = None

    if has_left:
        left_pos, left_x, ld = _compute_hierarchical_layout(node["left"], depth + 1, next_x)
        positions.update(left_pos)
        max_d = max(max_d, ld)

    if not has_left and not has_right:
        x = next_x[0]
        next_x[0] += 1.0
        positions[node_key] = (x, -depth, node, True)
        return positions, x, depth

    if has_right:
        right_pos, right_x, rd = _compute_hierarchical_layout(node["right"], depth + 1, next_x)
        positions.update(right_pos)
        max_d = max(max_d, rd)

    if left_x is not None and right_x is not None:
        curr_x = (left_x + right_x) / 2.0
    elif left_x is not None:
        curr_x = left_x + 0.5
    elif right_x is not None:
        curr_x = right_x - 0.5
    else:
        curr_x = next_x[0]
        next_x[0] += 1.0

    positions[node_key] = (curr_x, -depth, node, False)
    return positions, curr_x, max_d


def _format_node_text(
    node: Dict[str, Any],
    is_leaf: bool,
    feature_names: Optional[List[str]] = None,
    use_original_features: bool = True
) -> str:
    """
    Constructs a readable label for internal split nodes and leaf nodes.
    """
    if is_leaf:
        lines = []
        if "value" in node:
            val = node["value"]
            if isinstance(val, dict):
                if "rel_prob" in val:
                    lines.append(f"Prob: {val['rel_prob']:.2f}")
                if "sample_ids" in val:
                    lines.append(f"N: {len(val['sample_ids'])}")
                if "leaf_id" in val:
                    lines.append(f"ID: {str(val['leaf_id'])[:6]}")
            else:
                lines.append(f"Class: {val}")
        if "error" in node:
            try:
                err = float(node["error"])
                lines.append(f"Err: {err:.2f}" if err != int(err) else f"Err: {int(err)}")
            except Exception:
                lines.append(f"Err: {node['error']}")
        return "\n".join(lines) if lines else "Leaf"

    # Internal split node
    feat_id = extract_node_feature_id(node, use_original_features=use_original_features)
    if feat_id is not None:
        if feature_names is not None and 0 <= feat_id < len(feature_names):
            name = feature_names[feat_id]
            feat_text = f"Feat: {name} (X{feat_id})"
        else:
            feat_text = f"Feature {feat_id}"
    else:
        feat_text = "Split Node"

    lines = [feat_text]

    # Additional split details if split_obj is attached
    if "split_obj" in node:
        split_obj = node["split_obj"]
        if hasattr(split_obj, "vals") and split_obj.vals is not None:
            try:
                vals = split_obj.vals
                if hasattr(vals, "shape") and vals.shape != ():
                    val_str = f"[{', '.join([f'{float(v):.2f}' for v in vals.flatten()[:2]])}]"
                else:
                    val_str = f"{float(vals):.2f}"
                lines.append(f"thresh: {val_str}")
            except Exception:
                pass
    elif "feat" in node:
        lines.append(f"(bool split {node['feat']})")

    return "\n".join(lines)


def draw_single_tree(
    tree_dict: Dict[str, Any],
    ax: plt.Axes,
    title: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    use_original_features: bool = True
):
    """
    Renders a single decision tree on the given Matplotlib Axes.
    """
    positions, _, max_depth = _compute_hierarchical_layout(tree_dict)

    all_xs = [pos[0] for pos in positions.values()]
    all_ys = [pos[1] for pos in positions.values()]

    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)

    # 1. Draw directed edges from parent to children
    def draw_edges(parent_node: Dict[str, Any]):
        p_x, p_y, _, _ = positions[id(parent_node)]

        for direction, child_key, branch_label, color in [
            ("L", "left", "1 (L)", "#1E88E5"),
            ("R", "right", "0 (R)", "#E53935")
        ]:
            if child_key in parent_node and isinstance(parent_node[child_key], dict):
                child_node = parent_node[child_key]
                c_x, c_y, _, _ = positions[id(child_node)]

                # Connecting line
                ax.plot([p_x, c_x], [p_y, c_y], color="#757575", lw=1.5, zorder=1)

                # Branch decision label (1 / 0)
                mid_x = (p_x + c_x) / 2.0
                mid_y = (p_y + c_y) / 2.0
                ax.text(
                    mid_x, mid_y, branch_label,
                    fontsize=8, fontweight="bold", color=color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="none", alpha=0.85),
                    zorder=2
                )
                draw_edges(child_node)

    draw_edges(tree_dict)

    # 2. Draw nodes (split nodes vs leaf nodes)
    for node_id, (x, y, node_obj, is_leaf) in positions.items():
        text_str = _format_node_text(
            node_obj, is_leaf,
            feature_names=feature_names,
            use_original_features=use_original_features
        )

        if is_leaf:
            facecolor = "#E8F5E9"  # soft pastel green
            edgecolor = "#2E7D32"
            boxstyle = "square,pad=0.4"
        else:
            facecolor = "#E3F2FD"  # soft pastel blue
            edgecolor = "#1565C0"
            boxstyle = "round,pad=0.45"

        ax.text(
            x, y, text_str,
            fontsize=8.5,
            fontweight="medium",
            ha="center", va="center",
            bbox=dict(
                boxstyle=boxstyle,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.4,
                alpha=0.95
            ),
            zorder=3
        )

    # Configure axes limits and styling
    x_margin = max(0.6, (max_x - min_x) * 0.12)
    y_margin = 0.6
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
