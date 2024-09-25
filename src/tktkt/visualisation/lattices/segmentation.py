"""
The "segmentation lattice" or "segmentation trellis" is a directed acyclic graph for a string of n characters that
contains n+1 nodes placed on a straight horizontal line, where there is an arc between node i and node j iff i < j and
the string s[i:j] is in the vocabulary.

You can represent this lattice in multiple ways:
    - One big grid of (i,j) arc weights, where impossible arcs are indicated with a value of +/-INF.
    - Two equisize lists of length n+1, one including backpointers for each node j (i.e. each i that has an arc to j)
      and the other having the weights of those arcs.

The point of this file is NOT to create this lattice. It should have been computed elsewhere already.
"""
from typing import List
import numpy as np

from ...models.viterbi.framework import ViterbiStepScores
from ...util.strings import indent


def visualiseScoreGrid(score_grid: ViterbiStepScores, characters: str="",
                       do_numbered_states: bool=False, do_characters: bool=True, do_arc_labels: bool=True, do_alternate_arcs: bool=False):
    """
    :param score_grid: A characters x steps score grid like you would find in tktkt.models.viterbi.
    """
    do_characters = do_characters and characters

    N,K = score_grid.grid.shape
    assert N > 0
    if do_characters:
        assert len(characters) == N, f"Character dimension has {N} positions even though the string has {len(characters)} characters."

    # Draw nodes
    tikz = ""
    for i in range(N+1):
        tikz += f"\\node[state" + f", right of={i-1}"*(i != 0) + f"] ({i}) " + "{" + f"{i}"*do_numbered_states + "};\n"

    # Draw characters in between nodes
    if do_characters:
        tikz += "\\path\n"
        for i in range(N):
            tikz += f"    ({i}) --node[charstyle] " + "{" + characters[i] + "} " + f"({i+1})\n"
        tikz = tikz.rstrip() + ";\n"

    # Arcs
    tikz += "\\draw\n"
    for n in range(N):
        # First get the valid arcs.
        valid_ks = [k for k in range(K) if not np.isinf(score_grid.get(n,k))]

        # Then visualise on those
        arc_direction  =  "left" if not do_alternate_arcs or n % 2 == 0 else "right"  # "left" actually means "arc goes over" and "right" means "arc goes under".
        label_location = "above" if not do_alternate_arcs or n % 2 == 0 else "below"
        for i,k in enumerate(valid_ks):
            # Variations on "bend left" include:
            #   - suffixing "left" by "=NUMBERcm" to have the arc deviate from the baseline by a fixed distance
            #   - suffixing "left" by "=NUMBER" to have the arc leave at a unit circle angle in degrees (0 to 90 make sense)
            angle = 20 + i*min(10, (90-20)/len(valid_ks))
            tikz += f"    ({n}) edge[bend {arc_direction}={angle}, {label_location}] node[labelstyle] " "{" + f"{round(float(score_grid.get(n,k)),2)}"*do_arc_labels + "}" f" ({n+k+1})\n"
    tikz = tikz.rstrip() + ";\n"

    tikz = "\\begin{tikzpicture}\n" + indent(1, tikz) + "\\end{tikzpicture}\n"
    return tikz


# def visualiseBackpointers(backpointer_lists: List[List[int]], score_lists: List[List[float]], node_values: list, inter_node_values: list):
