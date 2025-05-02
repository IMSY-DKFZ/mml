# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

import logging
import warnings
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase, digits
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from openTSNE import TSNE

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.task_attributes import Keyword, TaskType
from mml.core.data_loading.task_struct import TaskStruct, undup_names
from mml.core.scripts.utils import TAG_SEP
from mml.core.visualization.utils import COLORS

logger = logging.getLogger(__name__)


@dataclass
class DistancePlotConfig:
    kamada_kawai: bool = True
    # kamada kawai parameters
    label_nodes: bool = False
    color_nodes: bool = True
    color_map: str = "jet"  # also 'hsv'
    node_size: int = 700
    plot_edges: bool = True
    label_edges: bool = False
    edge_thresh: float = 0.65  # 0 for no edges, -1 for all edges
    legend_size: int = 20  # <= 0 deactivates legend
    # colorization by...
    criteria: Optional[str] = None  # maybe either None, medical, task_type, target or domain
    # tsne parameters
    draw_centers: bool = False
    draw_cluster_labels: bool = False


cs = ConfigStore.instance()
cs.store(name="distance_plot_config", node=DistancePlotConfig)


def plot_task_embedding(
    distances: Union[pd.DataFrame, Path], structs: List[TaskStruct], plot_config: DistancePlotConfig, distance_key: str
) -> Path:
    if isinstance(distances, Path):
        distances = pd.read_csv(distances, index_col=0, header=0)
    all_tasks = distances.index.to_list()
    if distances.columns.to_list() != all_tasks:
        raise ValueError("Need all task distance pairs!")
    task_subset = [struct.name for struct in structs]
    if not set(task_subset).issubset(set(all_tasks)):
        raise ValueError(f"Found tasks: {all_tasks}, incompatible subset {task_subset}")
    to_drop = [t for t in all_tasks if t not in task_subset]
    distances = distances.drop(to_drop, axis="index").drop(to_drop, axis="columns")
    # assure same index for rows and columns
    all_tasks = distances.index.to_list()
    distances = distances[all_tasks]
    # shift distances so that negative values become feasible
    if (distances < 0).any().any():
        logger.warning("Found negative distance values, will shift distances to allow plotting.")
        smallest = distances.min().min()
        distances += 1.1 * (-smallest)
    plot_path = MMLFileManager.instance().construct_saving_path(obj=None, key="2D_" + distance_key.lower())
    color_map, legend_map = create_color_mapping(task_list=structs, criteria=plot_config.criteria)
    if plot_config.kamada_kawai:
        fig = plot_kamada_kawai(distances=distances, plot_cfg=plot_config, color_map=color_map, legend_map=legend_map)
    else:
        fig = plot_tsne(distances=distances, plot_cfg=plot_config, color_map=color_map, legend_map=legend_map)
    # saving the result
    fig.savefig(str(plot_path))
    fig.clf()
    plt.close()
    logger.info(f"Plotted distance graph @{plot_path}.")
    return plot_path


def plot_kamada_kawai(
    distances: pd.DataFrame,
    plot_cfg: DistancePlotConfig,
    color_map: Optional[Dict[str, int]] = None,
    legend_map: Optional[Dict[int, str]] = None,
    border_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Returns a figure with a task graph generated from provided task distances.

    :param Union[pd.DataFrame, Path] distances:
    :param Optional[Dict[str, int]] color_map:
    :param Optional[Dict[int, str]] legend_map:
    :param Optional[Dict[str, str]] border_map: (optional) give task nodes in the graph a certain border color
    :return:
    """

    all_tasks = distances.index.to_list()

    if plot_cfg.color_nodes:
        assert color_map is not None, "For coloring nodes provide a color_map"
        assert all([t in color_map for t in all_tasks]), f"color map only supports {color_map.keys()}"

    mean = distances.mask(np.eye(len(distances.index), dtype=bool)).replace([np.inf], np.nan).mean().mean()
    # check for symmetry
    symmetric = distances.equals(distances.T)
    # create graph and layout
    G = nx.Graph(distances) if symmetric else nx.DiGraph(distances)
    pos = nx.kamada_kawai_layout(G)
    # start figure for plotting
    fig = plt.figure(tight_layout=True, figsize=[12.0, 9.0])
    gs = matplotlib.gridspec.GridSpec(3, 1) if plot_cfg.label_nodes else matplotlib.gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0:2, 0])
    drawing_edges = (
        G.edges()
        if plot_cfg.edge_thresh == -1
        else [(s, t) for s, t in G.edges() if distances.at[s, t] < plot_cfg.edge_thresh * mean]
    )
    # color nodes by the color_map provided
    if color_map is not None:
        ordered_colors = [color_map.get(node, 0) for node in G.nodes()]
        all_colors = list(set(color_map.values()))
        # Color mapping
        color_map = plt.get_cmap(plot_cfg.color_map)
        normalizer = matplotlib.colors.Normalize(vmin=0, vmax=max(all_colors))
        scalar_map = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        # legend
        if plot_cfg.legend_size > 0:
            assert all([col in legend_map for col in all_colors]), "provide legend map for all entries of color_map"
            handles = []
            for col in all_colors:
                patch = matplotlib.patches.Patch(color=scalar_map.to_rgba(col), label=legend_map[col])
                handles.append(patch)
            ax.legend(handles=handles, fontsize=plot_cfg.legend_size)
        # draw colored nodes
        nx.draw_networkx(
            G,
            pos,
            cmap=color_map,
            vmin=0,
            vmax=max(all_colors),
            node_color=ordered_colors,
            arrows=False,
            edgelist=drawing_edges if plot_cfg.plot_edges else None,
            node_size=plot_cfg.node_size,
            with_labels=False,
            width=1.0 if plot_cfg.plot_edges else 0.0,
            edgecolors=[border_map.get(node, 0) for node in G.nodes()] if border_map else None,
            linewidths=1.75,
        )
    else:
        # color all nodes equal
        nx.draw_networkx(G, pos, with_labels=False, width=1.0 if plot_cfg.plot_edges else 0.0)
    unmoded_label_dict = {}
    if plot_cfg.label_nodes:
        ordered_node_labels = ascii_uppercase + ascii_lowercase + digits
        unmoded_node_names = undup_names(list(G.nodes()))
        # check if enough symbols to label every node in the Graph
        if len(ordered_node_labels) < len(set(unmoded_node_names)):
            logger.error(
                f"Was unable to label nodes, was given {len(set(unmoded_node_names))} node names, but only "
                f"support up to {len(ordered_node_labels)} node names. Skipped node labeling."
            )
        else:
            # define for every unmoded node some label
            for ix, unmoded in enumerate(list(set(unmoded_node_names))):
                unmoded_label_dict[unmoded] = ordered_node_labels[ix]
            # now define labels for each node
            node_labels = {}
            for ix, node in enumerate(G.nodes):
                node_labels[node] = unmoded_label_dict[unmoded_node_names[ix]]
            nx.draw_networkx_labels(G, pos, labels=node_labels)
    if plot_cfg.label_edges:
        edge_labels = {}
        for edge in drawing_edges:
            edge_labels[(edge[0], edge[1])] = "{:.2f}".format(distances.at[edge[0], edge[1]])
        position = 0.5 if symmetric else 0.3
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=position)
    # turn on / off surrounding box of the graph and set width of boxlines
    ax.axis("on")
    for orient in ["top", "right", "left", "bottom"]:
        ax.spines[orient].set_linewidth(5.0)
    if plot_cfg.label_nodes:
        # switch to labels part of the plot (lower figure)
        ax = fig.add_subplot(gs[2, 0])
        text_lines = ["{}: {}".format(v, k) for k, v in unmoded_label_dict.items()]
        lines_per_column = 15
        num_text_cols = int(len(text_lines) / lines_per_column) + 1
        for text_column in range(num_text_cols):
            ax.text(
                text_column * (1 / num_text_cols),
                0.5,
                "Datasets:\n"
                + "\n".join(text_lines[text_column * lines_per_column : (text_column + 1) * lines_per_column])
                + "\n",
                ha="center",
                va="center",
                wrap=True,
            )
        ax.axis("off")
    # background color of the figure
    fig.set_facecolor("w")
    return fig


def create_color_mapping(
    task_list: List[TaskStruct],
    criteria: Optional[str] = None,
    tag_groups: Optional[Dict[str, List[Keyword]]] = None,
    task_clusters: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Creates color map and legend map based on either criteria, self-defined tag_groups or task clusters
     to cluster tasks.

    :param task_list: tasks to be color selected
    :param criteria: one of the predefined cluster methods (medical, domain, task_type, tagged, size)
    :param tag_groups: self provided clustering of keywords (keywords within a groups are OR connected)
    :param task_clusters: self provided clustering of tasks
    :return: tuple with color_map and legend_map as required by plot_2D
    """
    # compatibility with hydra config
    if criteria in ["None", "none"]:
        warnings.warn(
            DeprecationWarning(
                "Using string None as argument to distance plotting criteria is deprecated. "
                "In hydra CLI use null instead."
            )
        )
        criteria = None
    # input values check
    assert sum([criteria is None, tag_groups is None, task_clusters is None]) >= 2, (
        "provide at most one of criteria, tag_groups or task_clusters"
    )
    if sum([criteria is None, tag_groups is None, task_clusters is None]) == 3:
        # default behaviour in case nothing was given
        criteria = "domain"
    if criteria is not None:
        # define tag groups
        if criteria == "medical":
            tag_groups = {"Medical": [Keyword.MEDICAL]}
        elif criteria == "domain":
            tag_groups = {
                str(tag.value).capitalize(): [tag]
                for tag in [
                    Keyword.DERMATOSCOPY,
                    Keyword.LARYNGOSCOPY,
                    Keyword.GASTROSCOPY_COLONOSCOPY,
                    Keyword.LAPAROSCOPY,
                    Keyword.NATURAL_OBJECTS,
                    Keyword.HANDWRITINGS,
                    Keyword.CATARACT_SURGERY,
                    Keyword.FUNDUS_PHOTOGRAPHY,
                    Keyword.MRI_SCAN,
                    Keyword.X_RAY,
                    Keyword.CT_SCAN,
                    Keyword.CLE,
                    Keyword.ULTRASOUND,
                    Keyword.CAPSULE_ENDOSCOPY,
                ]
            }
        elif criteria == "target":
            tag_groups = {
                str(tag.value).capitalize(): [tag]
                for tag in [
                    Keyword.ENDOSCOPIC_INSTRUMENTS,
                    Keyword.ANATOMICAL_STRUCTURES,
                    Keyword.TISSUE_PATHOLOGY,
                    Keyword.IMAGE_ARTEFACTS,
                ]
            }
        elif criteria == "task_type":
            task_clusters = {
                task_type.value: [task.name for task in task_list if task.task_type == task_type]
                for task_type in list(TaskType)
            }
        elif criteria == "tagged":
            plain = [task.name for task in task_list if " " not in task.name and TAG_SEP not in task.name]
            task_clusters = {"plain": plain, "modified": [task.name for task in task_list if task.name not in plain]}
        elif criteria == "size":
            task_clusters = {
                "small": [t.name for t in task_list if t.num_samples < 1000],
                "medium": [t.name for t in task_list if 1000 < t.num_samples < 10000],
                "large": [t.name for t in task_list if t.num_samples > 10000],
            }
        else:
            raise ValueError(f"criteria {criteria} is an invalid value")
    if tag_groups is not None:
        # convert tag groups to task clusters
        task_clusters = {
            group_name: [task.name for task in task_list if any([tag in task.keywords for tag in group_tags])]
            for group_name, group_tags in tag_groups.items()
        }
    # check task clusters - completeness and uniqueness
    for (name_a, cluster_a), (name_b, cluster_b) in combinations(task_clusters.items(), 2):
        for duplicate in set(cluster_a).intersection(cluster_b):
            logger.warning(f"Found duplicate task {duplicate} in {name_a} and {name_b}, will remove from {name_b}.")
            cluster_b.remove(duplicate)
    others = set([task.name for task in task_list]).difference(set(chain(*task_clusters.values())))
    if len(others) > 0:
        task_clusters["Other"] = list(others)
    # produce mappings
    clusters = sorted(list(task_clusters.keys()))
    legend_map = {ix: cluster_name for ix, cluster_name in enumerate(clusters)}
    task_mapping = {name: cluster for cluster in task_clusters.keys() for name in task_clusters[cluster]}
    color_map = {task.name: clusters.index(task_mapping[task.name]) for task in task_list}
    return color_map, legend_map


def plot_tsne(
    distances: Union[pd.DataFrame, Path],
    plot_cfg: DistancePlotConfig,
    color_map: Optional[Dict[str, int]] = None,
    legend_map: Optional[Dict[int, str]] = None,
) -> plt.Figure:
    """
    Adapted from https://github.com/pavlin-policar/openTSNE/blob/master/examples/utils.py

    :param distances:
    :param plot_cfg:
    :param color_map:
    :param legend_map:
    :return:
    """
    # generate TSNE embedding
    embedding = TSNE(metric="precomputed", random_state=42).fit(distances.to_numpy())
    # base plot
    fig, ax = plt.subplots(figsize=(12, 9))
    y = [color_map[task_name] for task_name in distances.index]
    classes = np.unique(list(color_map.values()))
    colors = {ix: COLORS[ix % len(COLORS)] for ix in classes}
    point_colors = list(map(colors.get, y))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=point_colors, rasterized=True, s=float(plot_cfg.node_size))
    # plot mediods
    if plot_cfg.draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(embedding[mask, :2], axis=0))
        centers = np.array(centers)
        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=float(plot_cfg.node_size) * 2, alpha=1, edgecolor="k"
        )
        # Draw mediod labels
        if plot_cfg.draw_cluster_labels:
            for idx in classes:
                ax.text(
                    centers[idx, 0], centers[idx, 1] + 2.2, legend_map[idx], fontsize=6, horizontalalignment="center"
                )
    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")
    # legend
    if plot_cfg.legend_size > 0:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[ix],
                ms=10,
                alpha=1,
                linewidth=0,
                label=legend_map[ix],
                markeredgecolor="k",
            )
            for ix in classes
        ]
        ax.legend(handles=legend_handles, fontsize=plot_cfg.legend_size)
    # background color of the figure
    fig.set_facecolor("w")
    return fig
