import sys, os
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Union, Optional, Tuple, Literal, Callable
from functools import partial
from IPython.display import Image, display
from tqdm import tqdm
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, ActivationCache
import plotly.io as pio
from transformers import pipeline
from pprint import pprint
from einops import einsum

pio.renderers.default = "colab"

device = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("WARNING: Running on CPU. Did you remember to set your Colab accelerator to GPU?")

model_name = 'gpt2-large'
model = HookedTransformer.from_pretrained(model_name, device=device)
model.cfg.use_attn_in = True
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

"""# **Reading and Modifying Dataset**"""

from typing import List, Dict, Union, Tuple, Literal, Optional, Set
from collections import defaultdict
from pathlib import Path
import json
import heapq

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np
import pygraphviz as pgv

from eap.visualization import EDGE_TYPE_COLORS, generate_random_color


df=pd.read_csv("")

class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs,
    while the out_hook gets its outputs.
    """
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple, qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f'Node({self.name})'

    def __hash__(self):
        return hash(self.name)

class LogitNode(Node):
    def __init__(self, n_layers:int):
        name = 'logits'
        index = slice(None)
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index)

class MLPNode(Node):
    def __init__(self, layer: int):
        name = f'm{layer}'
        index = slice(None)
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int):
        name = f'a{layer}.h{head}'
        self.head = head
        index = (slice(None), slice(None), head)
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, [f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    def __init__(self):
        name = 'input'
        index = slice(None)
        super().__init__(name, 0, '', "blocks.0.hook_resid_pre", index)

class Edge:
    name: str
    parent: Node
    child: Node
    hook: str
    index: Tuple
    score : Optional[float]
    in_graph: bool
    def __init__(self, parent: Node, child: Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent
        self.child = child
        self.qkv = qkv
        self.score = None
        self.in_graph = True
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook
    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF00FF"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return f'Edge({self.name})'

    def __hash__(self):
        return hash(self.name)

class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    n_forward: int
    n_backward: int
    cfg: HookedTransformerConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        edge = Edge(parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)

    def forward_index(self, node:Node, attn_slice=True):
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
            # raise ValueError(f"No forward for logits node")
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")


    def backward_index(self, node:Node, qkv=None, attn_slice=True):
        if isinstance(node, InputNode):
            raise ValueError(f"No backward for input node")
        elif isinstance(node, LogitNode):
            return -1
        elif isinstance(node, MLPNode):
            return (node.layer) * (3 * self.cfg['n_heads'] + 1) + 3 * self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            i = node.layer * (3 * self.cfg['n_heads'] + 1) + ('qkv'.index(qkv) * self.cfg['n_heads'])
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def scores(self, absolute=False, nonzero=False, in_graph=False, sort=True):
        s = [edge.score for edge in self.edges.values() if edge.score != 0 and (edge.in_graph or not in_graph)] if nonzero else [edge.score for edge in self.edges.values()]
        s = torch.tensor(s)
        if absolute:
            s = s.abs()
        return torch.sort(s).values if sort else s

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())

    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())

    def apply_threshold(self, threshold: float, absolute: bool):
        threshold = float(threshold)
        for node in self.nodes.values():
            node.in_graph = True

        for edge in self.edges.values():
            edge.in_graph = abs(edge.score) <= threshold if absolute else edge.score <= threshold

    def apply_greedy(self, n_edges, reset=True, absolute: bool=False):
        if reset:
            for node in self.nodes.values():
                node.in_graph = False
            for edge in self.edges.values():
                edge.in_graph = False
            self.nodes['logits'].in_graph = True

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph], key = lambda edge: abs_id(edge.score), reverse=True)

        edges = heapq.merge(candidate_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key = lambda edge: abs_id(edge.score), reverse=True)
                edges = heapq.merge(edges, parent_parent_edges, key = lambda edge: abs_id(edge.score), reverse=True)

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        self.nodes['logits'].in_graph = any(parent_edge.in_graph for parent_edge in self.nodes['logits'].parent_edges)

        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue

            if any(child_edge.in_graph for child_edge in node.child_edges) :
                node.in_graph = True
            else:
                if prune_childless:
                    node.in_graph = False
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph = False
                else:
                    if any(child_edge.in_graph for child_edge in node.child_edges):
                        node.in_graph = True
                    else:
                        node.in_graph = False

        if prune_parentless:
            for node in self.nodes.values():
                if not isinstance(node, InputNode) and node.in_graph and not any(parent_edge.in_graph for parent_edge in node.parent_edges):
                    node.in_graph = False
                    for child_edge in node.child_edges:
                        child_edge.in_graph = False


    @classmethod
    def from_model(cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict]):
        graph = Graph()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp}
        else:
            graph.cfg = model_or_config

        input_node = InputNode()
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg['n_heads'])]
            mlp_node = MLPNode(layer)

            for attn_node in attn_nodes:
                graph.nodes[attn_node.name] = attn_node
            graph.nodes[mlp_node.name] = mlp_node

            if graph.cfg['parallel_attn_mlp']:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in 'qkv':
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)

                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in 'qkv':
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg['n_layers'])
        for node in residual_stream:
            graph.add_edge(node, logit_node)

        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1)
        graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1

        return graph


    def to_json(self, filename):
        # non serializable info
        d = {'cfg':self.cfg, 'nodes': {str(name): bool(node.in_graph) for name, node in self.nodes.items()}, 'edges':{str(name): {'score': float(edge.score), 'in_graph': bool(edge.in_graph)} for name, edge in self.edges.items()}}
        with open(filename, 'w') as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            d = json.load(f)
        g = Graph.from_model(d['cfg'])
        for name, in_graph in d['nodes'].items():
            g.nodes[name].in_graph = in_graph

        for name, info in d['edges'].items():
            g.edges[name].score = info['score']
            g.edges[name].in_graph = info['in_graph']

        return g

    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (set(self.edges.keys()) == set(other.edges.keys()))
        if not keys_equal:
            return False

        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False

        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(edge.score, other.edges[name].score):
                return False
        return True

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.3,
        layout: str="dot",
        seed: Optional[int] = None
    ) -> pgv.AGraph:
        """
        Colorscheme: a cmap colorscheme
        """
        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        #colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                if node.name in {'input'}:
                  clr='yellowgreen'
                elif node.name in {'logits'}:
                  clr='gold'
                elif node.name in {'m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11'}:
                  clr = 'plum'
                else:
                  clr='sandybrown'
                g.add_node(node.name,
                        fillcolor=clr,
                        color="black",
                        style="filled, rounded",
                        shape="box",
                        fontname="Helvetica",
                        )

        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if edge.score is None else edge.score
                g.add_edge(edge.parent.name,
                        edge.child.name,
                        penwidth=str(max(minimum_penwidth, score) * 2),
                        color='black',
                        )
        return g

"""## **Rest of the functions**"""

g = Graph.from_model(model)
g1 = Graph.from_model(model)

print(list(g.nodes.items())[:15])
print(list(g.edges.items())[:15])

print(f'Total No. of Nodes in Model: {len(list(g.nodes.items())[:])}')
print(f'Total No. of edges in Model: {len(list(g.edges.items())[:])}')


def prob_diff(sentence, logits: torch.Tensor, top_k, loss=False, mean=False):
    MB_Probs = 0
    FB_Probs = 0
    GN_Probs = 0
    k=top_k
    token_ids = torch.argmax(logits, dim=-1)
    token_ids = token_ids.cpu()
    token_ids_list = token_ids.tolist()[0]  # Convert tensor to list
    input_sentence = model.tokenizer.decode(token_ids_list)  # Corrected line
    probs = torch.softmax(logits[:,-1], dim=-1)
    probs, next_tokens = torch.topk(probs[-1], k)
    results = []
    for i, (prob, token_id) in enumerate(zip(probs,next_tokens)):
        token = model.tokenizer.decode(token_id.item())
        predicted = sentence[0] + token
        bias = gender_terms.get(token.lower(), 0)
        if bias == 1:
          MB_Probs += prob.sum()
        elif bias == -1:
          FB_Probs += prob.sum()
        else:
          GN_Probs += prob.sum()

    results.append(MB_Probs - GN_Probs)
    results = torch.stack(results)
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results



metric = prob_diff

def evaluate_baseline(model: HookedTransformer, dataset, top_k: int, metrics: List[Callable[[Tensor], Tensor]]):
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False

    results = [[] for _ in metrics]
    for sentence, corrupted in tqdm(dataset):
        with torch.inference_mode():
            logits = model(sentence)
        for j, metric in enumerate(metrics):
            r = metric(sentence,logits, top_k).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[j].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def evaluate_graph(model: HookedTransformer, graph: Graph, dataset,top_k: int, metrics: List[Callable[[Tensor], Tensor]], prune:bool=True):
    """
    Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.
    """
    # Pruning the Graph: If prune is True, it prunes the graph by removing childless and parentless nodes.
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    # Check for Empty Circuit: Sets empty_circuit to True if the 'logits' node is not in the graph.
    empty_circuit = not graph.nodes['logits'].in_graph

    # Forward Hook Names: Collects the output hooks of parent nodes from all edges in the graph.
    # Forward Filter: Creates a filter function to check if a given hook name is in fwd_names.
    fwd_names = {edge.parent.out_hook for edge in graph.edges.values()}
    fwd_filter = lambda x: x in fwd_names

    # Get Caching Hooks: Retrieves the corrupted and mixed forward caches and hooks from the model using the forward filter.
    corrupted_fwd_cache, corrupted_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)
    mixed_fwd_cache, mixed_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)

    # Nodes in Graph: Collects all nodes in the graph that are not of type InputNode and are part of the graph (in_graph is True).
    nodes_in_graph = [node for node in graph.nodes.values() if node.in_graph if not isinstance(node, InputNode)]

    """For each node in the graph, construct its input (in the case of attention heads, multiple inputs) by corrupting the incoming edges that are not in the circuit.
       We assume that the corrupted cache is filled with corresponding corrupted activations, and that the mixed cache contains the computed activations from preceding nodes in this forward pass."""
    # Input Construction Hook: Defines a nested function make_input_construction_hook that creates an input_construction_hook.
    # Inner Function input_construction_hook: Iterates over the parent edges of a node.
    # If the qkv attribute of an edge does not match the provided qkv parameter, it skips the edge.
    # If the edge is not part of the graph (in_graph is False), it modifies the activations by replacing values from the mixed forward cache with those from the corrupted forward cache.
    # Return: Returns the input_construction_hook function.
    def make_input_construction_hook(node: Node, qkv=None):
        def input_construction_hook(activations, hook):
            for edge in node.parent_edges:
                if edge.qkv != qkv:
                    continue

                parent:Node = edge.parent
                if not edge.in_graph:
                    activations[edge.index] -= mixed_fwd_cache[parent.out_hook][parent.index]
                    activations[edge.index] += corrupted_fwd_cache[parent.out_hook][parent.index]
            return activations
        return input_construction_hook

    # Create Input Construction Hooks: Iterates over the nodes in the graph to create input construction hooks.
    # InputNode: Skips if the node is an InputNode.
    # LogitNode or MLPNode: Adds a hook using make_input_construction_hook without qkv.
    # AttentionNode: Adds hooks for each of 'q', 'k', and 'v' inputs.
    # Invalid Node: Raises an error if the node type is not recognized.
    input_construction_hooks = []
    for node in nodes_in_graph:
        if isinstance(node, InputNode):
            pass
        elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
            input_construction_hooks.append((node.in_hook, make_input_construction_hook(node)))
        elif isinstance(node, AttentionNode):
            for i, letter in enumerate('qkv'):
                input_construction_hooks.append((node.qkv_inputs[i], make_input_construction_hook(node, qkv=letter)))
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    # and here we actually run / evaluate the model
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]

    for sentence, corrupted in tqdm(dataset):
        sens = [sentence, corrupted]
        sens = [str(s) for s in sens]
        max_length = max(len(model.tokenizer.tokenize(s)) for s in sens)
        padded_sentences = [model.tokenizer.encode(s, padding='max_length', max_length=max_length, return_tensors='pt', add_special_tokens=True) for s in sens]
        s1 = padded_sentences[0]
        s2 = padded_sentences[1]
        clean = model.tokenizer.decode(s1[0])
        corrupted_dash = model.tokenizer.decode(s2[0])

        with torch.inference_mode():
            with model.hooks(corrupted_fwd_hooks):
                corrupted_logits = model(corrupted_dash)

            with model.hooks(mixed_fwd_hooks + input_construction_hooks):
                if empty_circuit:
                    # if the circuit is totally empty, so is nodes_in_graph
                    # so we just corrupt everything manually like this
                    logits = model(corrupted_dash)
                else:
                    logits = model(clean)
        for i, metric in enumerate(metrics):
            r = metric(sentence,logits).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

"""# **Baseline Scoring**"""

### Baseline score for Countries where gpt2 shows more positive sentiment
baseline_pos_dataset = evaluate_baseline(model, MB_dataset,10, metric).mean()
print((f"\n Baseline performance of positive dataset: {baseline_pos_dataset}"))

### Baseline score for Countries where gpt2 shows more negative sentiment
baseline_neg_dataset = evaluate_baseline(model, FB_dataset,10, metric).mean()
print((f"\n Baseline performance of negative dataset: {baseline_neg_dataset}"))

### Graph_Baseline score for Countries where gpt2 shows more positive sentiment
graph_baseline_pos = evaluate_graph(model, g, MB_dataset,10, metric).mean()
print((f"\n Graph_Baseline performance for positive dataset: {graph_baseline_pos}"))

### Graph_Baseline score for Countries where gpt2 shows more negative sentiment
graph_baseline_neg = evaluate_graph(model, g, FB_dataset,10, metric).mean()
print((f"\n Graph_Baseline performance for Negative dataset: {graph_baseline_neg}"))

"""# **Edge Attribution Patching**

## **EAP Attribute Function**
"""

def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=True)
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores):
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach()
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e

    def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        grads = gradients.detach()
        try:
            if isinstance(fwd_index, slice):
                fwd_index = fwd_index.start
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(activation_difference[:, :, :fwd_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)
            scores[:fwd_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), grads.size())
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        fwd_index =  graph.forward_index(node)
        if not isinstance(node, LogitNode):
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, fwd_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, bwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

######                #######
#####  Edit from here  #######
#####                 #######

def get_scores(model: HookedTransformer, graph: Graph, dataset, metric: Callable[[Tensor], Tensor]):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    for sentence, corrupted in tqdm(dataset):
        sens = [sentence, corrupted]
        sens = [str(s) for s in sens]
        max_length = max(len(model.tokenizer.tokenize(s)) for s in sens)
        padded_sentences = [model.tokenizer.encode(s, padding='max_length', max_length=max_length, return_tensors='pt', add_special_tokens=True) for s in sens]
        s1 = padded_sentences[0]
        s2 = padded_sentences[1]
        clean = model.tokenizer.decode(s1[0])
        corrupted_dash = model.tokenizer.decode(s2[0])

        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted_dash)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean)
            label = torch.tensor(0, device='cuda', dtype=model.cfg.dtype)
            metric_value = metric(sentence,logits)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_ig(model: HookedTransformer, graph: Graph, dataset, metric: Callable[[Tensor], Tensor], steps=30):
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)

    total_items = 0
    for sentence, corrupted in tqdm(dataset):
        sens = [sentence, corrupted]
        sens = [str(s) for s in sens]
        max_length = max(len(model.tokenizer.tokenize(s)) for s in sens)
        padded_sentences = [model.tokenizer.encode(s, padding='max_length', max_length=max_length, return_tensors='pt', add_special_tokens=True) for s in sens]
        s1 = padded_sentences[0]
        s2 = padded_sentences[1]
        clean = model.tokenizer.decode(s1[0])
        corrupted_dash = model.tokenizer.decode(s2[0])

        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_dash)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_clean + (k / steps) * (input_activations_corrupted - input_activations_clean)
                new_input.requires_grad = True
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean)
                label = torch.tensor(0, device='cuda', dtype=model.cfg.dtype)
                metric_value = metric(sentence,logits)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores

allowed_aggregations = {'sum', 'mean', 'l2'}

### Done upto here #########


def attribute(model: HookedTransformer, graph: Graph, dataset, metric: Callable[[Tensor], Tensor], aggregation='sum', integrated_gradients: Optional[int]=None):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')


    if integrated_gradients is None:
        scores = get_scores(model, graph, dataset, metric)
    else:
        assert integrated_gradients > 0, f"integrated_gradients gives positive # steps (m), but got {integrated_gradients}"
        scores = get_scores_ig(model, graph, dataset, metric, steps=integrated_gradients)

        if aggregation == 'mean':
            scores /= model.cfg.d_model
        elif aggregation == 'l2':
            scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    scores = scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]

"""## **EAP and finding best scoring edges for positive Dataset**"""

attribute(model, g, MB_dataset, partial(metric, loss=True, mean=True))

# include all edges whose absolute score is >= the 4th greatest absolute score
scores = g.scores(absolute=True)
# using a greedy search over the graph, starting from the logits, add in the highest-scoring edges (non-absolute)
#g.apply_greedy(2000)

#g.apply_threshold(scores[-30], absolute=True)
# using a greedy search over the graph, starting from the logits, add in the 400 highest-scoring edges (non-absolute)
g.apply_greedy(90)
g.prune_dead_nodes()

# Checking the saturation in edge scores
No_of_top_scoring_edge_pos = np.arange(0, 810703 , 400)
least_score_saturated_pos = []
for num in No_of_top_scoring_edge_pos:
  least_score_saturated_pos.append(scores[num])


plt.figure(figsize=(12,6))
plt.plot(No_of_top_scoring_edge_pos, least_score_saturated_pos, marker='o')
plt.xlabel('Number of Edges from lowest to highest scoring')
plt.ylabel('EAP Score of edges')
plt.title('GPT2 Large (Male Bias GSS1)')
plt.xlim(0)
plt.ylim(0)
# plt.savefig('EAP score distribution for GPT2-small-GSS1-male.jpg')
plt.show()

# Get the remaining edges as a list
remaining_edges_pos = list(g.edges.items())

# Sort edges by their score (descending order)
remaining_edges_pos.sort(key=lambda x: abs(x[1].score), reverse=True)

# Print the top 10 edges
for i, (edge_id, edge) in enumerate(remaining_edges_pos[:3]):
    print(edge)
    print(f"  Score: {abs(edge.score)}")

top400_edges_pos=[]
for i, (edge_id,edge) in enumerate(remaining_edges_pos[:1000]):
  top400_edges_pos.append(str(edge_id))
filename = "top1000_edges_gpt2large_gss1_MB.txt"
with open(filename, "w") as file:
    for item in top400_edges_pos:
        file.write(f"{item}\n")

# comparing evaluate graph vs baseline
ablated_nscores_pos=[graph_baseline_pos]
intervalss=[10,50,100,150,200,250,300,350,400,450]
Edge_names_pos = []
for i, (edge_id,edge) in enumerate(remaining_edges_pos[:500]):
  Edge_names_pos.append(str(edge_id))
for idx in intervalss:
  for i in range(idx):
    g.edges[Edge_names_pos[i]].in_graph=False

  ablated_nscores_pos.append(evaluate_graph(model,g,MB_dataset,10,metric).mean())

  for j in range(idx):
    g.edges[Edge_names_pos[j]].in_graph=True


intervalss1=[0,10,50,100,150,200,250,300,350,400,450]
plt.figure(figsize=(12,6))
plt.plot(intervalss1,ablated_nscores_pos,marker='o',color='green')
plt.plot(intervalss1,[baseline_pos_dataset]*11,color='red')
plt.xlabel('Number of Top scoring edges ablated')
plt.ylabel('Evaluate Graph Score')
plt.title('GPT2 Large Male GSS1')
plt.xlim(0)
#plt.ylim(0)
plt.show()

# Ablating the top 10 edges in the graph and recording the evaluate graph scores(First we ablate top one edges, then top two edges and so on upto top 10 edges )
ablated_scores_pos=[graph_baseline_pos]
Edge_names_pos = []
for i, (edge_id,edge) in enumerate(remaining_edges_pos[:10]):
  Edge_names_pos.append(str(edge_id))
for index, _ in enumerate(Edge_names_pos):
  for i in range(index+1):
    g.edges[Edge_names_pos[i]].in_graph=False

  ablated_scores_pos.append(evaluate_graph(model,g,MB_dataset,10,metric).mean())

  for j in range(index+1):
    g.edges[Edge_names_pos[j]].in_graph=True

abs_ablated_scores_pos=[]
for i, _ in enumerate(ablated_scores_pos):
  abs_ablated_scores_pos.append(abs(ablated_scores_pos[i]))

print(abs_ablated_scores_pos)

plt.figure(figsize=(12,6))
plt.plot(list(range(0,11)),abs_ablated_scores_pos,marker='o',color='green')
plt.xlabel('Number of Top scoring edges ablated')
plt.ylabel('Evaluate Graph Score')
plt.title('Evaluate Graph Scores versus Number of Top Scoring edges ablated (for Male Bias Dataset)')
plt.xlim(0)
plt.ylim(0)
plt.show()

g3=g
g3.apply_threshold(scores[-10], absolute=True)

g3.prune_dead_nodes()

print(f"The auto-circuit has {g3.count_included_edges()} edges")

results = evaluate_graph(model, g3, MB_dataset,10, metric).mean()
print(f"\nGraph_baseline_positive performance: {graph_baseline_pos}. positive_ablated_circuit performance: {results}")
print(f"The score difference between original and ablated circuits = {graph_baseline_pos - results}")


"""## **Comparing Positive and Negative dataset results on the Graph**"""

plt.figure(figsize=(14,8))
plt.plot(list(range(0,11)),ablated_scores_pos,marker='o',color='violet',label='Ablated Scores(MB dataset)')
plt.plot(list(range(0,11)),ablated_scores_neg,marker='o',color='green',label='Ablated Scores(FB dataset)')
plt.plot(list(range(0,11)),[baseline_pos_dataset]*11,color='Red', label='Baseline(MB dataset)')
plt.plot(list(range(0,11)),[baseline_neg_dataset]*11,color='Orange', label='Baseline(FB dataset)')
plt.xlabel('Number of Top Scoring edges ablated', fontsize=16, fontweight='bold')
plt.ylabel('Evaluate Graph Score', fontsize=16, fontweight='bold')
plt.title('GPT2-Large: GSS1', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlim(0)

plt.legend(loc='best', fontsize=14)
plt.tight_layout()

plt.savefig('GPT2-Large: GSS1.pdf',format='pdf')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(intervalss1,ablated_nscores_neg,marker='o',color='green',label='Ablated Scores(FB dataset)')
plt.plot(intervalss1,ablated_nscores_pos,marker='o',color='blue',label='Ablated Scores(MB dataset)')
plt.plot(intervalss1,[baseline_neg_dataset]*11,color='red',label='Baseline(FB dataset)')
plt.plot(intervalss1,[baseline_pos_dataset]*11,color='pink',label='Baseline(MB dataset)')
plt.xlabel('Number of Top scoring edges ablated', fontsize=16, fontweight='bold')
plt.ylabel('Evaluate Graph Score', fontsize=16, fontweight='bold')
plt.title('GPT2 Large GSS1: Baseline vs. evaluate graph', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.legend(loc='best')
plt.xlim(0)
#plt.ylim(0)
plt.savefig('GPT2 Large GSS1: Baseline vs. evaluate graph.pdf',format='pdf')
plt.show()


"""# **Relation between Top-K and Baseline Score**"""

def prob_diff(sentence, logits: torch.Tensor, top_k, loss=False, mean=False):
    MB_Probs = 0
    FB_Probs = 0
    GN_Probs = 0
    k=top_k
    token_ids = torch.argmax(logits, dim=-1)
    token_ids = token_ids.cpu()
    token_ids_list = token_ids.tolist()[0]  # Convert tensor to list
    input_sentence = model.tokenizer.decode(token_ids_list)  # Corrected line
    probs = torch.softmax(logits[:,-1], dim=-1)
    probs, next_tokens = torch.topk(probs[-1], k)
    results = []
    for i, (prob, token_id) in enumerate(zip(probs,next_tokens)):
        token = model.tokenizer.decode(token_id.item())
        predicted = sentence[0] + token
        bias = gender_terms.get(token.lower(), 0)
        if bias == 1:
          MB_Probs += prob.sum()
        elif bias == -1:
          FB_Probs += prob.sum()
        else:
          GN_Probs += prob.sum()

    results.append(MB_Probs - GN_Probs)
    results = torch.stack(results)
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results



top_k_set = [5, 10, 15, 20, 25, 30]
baseline_scores_pos = []
for top_k in top_k_set:
  baseline_pos = evaluate_baseline(model, MB_dataset, top_k, metric).mean()
  baseline_scores_pos.append(baseline_pos)
  print(f"  For top_k = {top_k}, the baseline score is: {baseline_pos}\n")

print(baseline_scores_pos)

top_k_set = [5, 10, 15, 20, 25, 30]
baseline_scores_neg = []
for top_k in top_k_set:
  baseline_neg = evaluate_baseline(model, FB_dataset, top_k, metric).mean()
  baseline_scores_neg.append(baseline_neg)
  print(f"  For top_k = {top_k}, the baseline score is: {baseline_neg}\n")

print(baseline_scores_neg)

plt.figure(figsize=(12, 6))
plt.plot(top_k_set, baseline_scores_pos, marker='o',color="blue",label='MB dataset')
plt.plot(top_k_set, baseline_scores_neg, marker='o',color="green",label='FB dataset')
plt.xlabel('Top-K', fontsize=16, fontweight='bold')
plt.ylabel('Baseline Scores', fontsize=16, fontweight='bold')
plt.title('Baseline Score vs. Top-K for GPT1-Large GSS1', fontsize=18, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')  # Larger and bold font for ticks
plt.yticks(fontsize=14, fontweight='bold')

plt.xlim(5)
plt.legend(loc='best')
plt.savefig('GPT2-Large GSS1: Baseline Score vs. Top-K.pdf',format='pdf')
plt.show()

arrays = {
    "baseline_pos_dataset": baseline_pos_dataset,
    "baseline_neg_dataset": baseline_neg_dataset,
    "graph_baseline_pos": graph_baseline_pos,
    "graph_baseline_neg": graph_baseline_neg,
    "least_score_saturated_pos": least_score_saturated_pos,
    "ablated_nscores_neg": ablated_nscores_neg,
    "abs_ablated_scores_pos": abs_ablated_scores_pos,
    "least_score_saturated_neg": least_score_saturated_neg,
    "ablated_nscores_neg": ablated_nscores_neg,
    "abs_ablated_scores_neg": abs_ablated_scores_neg,
    "baseline_scores_neg": baseline_scores_neg,
    "baseline_scores_pos": baseline_scores_pos
}


file_path = "GPT2Large_GSS1.txt"

with open(file_path, 'w') as f:
    for var_name, array in arrays.items():
        f.write(f"{var_name}: {str(array)}\n")
