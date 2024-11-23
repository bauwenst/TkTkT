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
from typing import List, Optional
import numpy as np

from ...models.random.graph import SegmentationGraph
from ...models.viterbi.framework import ViterbiStepScores
from ...util.strings import indent


class LinearDAGToTikz:

    def tikzPreamble(self) -> str:
        return r"""
\tikzset{
    automatastyle/.style={
        shorten >=1pt, on grid, auto,
        ->, >=Stealth, 
        every state/.style={thick, minimum size=1em},
        initial text =,  % There is an extra invisible node that enters into the state marked "initial", and normal that node carries the text "start".
    },
    charstyle/.style={
        anchor=center, 
        font=\scshape
    }, 
    labelstyle/.style={
        scale=0.67
    }
}
        """

    def _renderNodes(self, number_of_nodes: int, node_values: list=None, inter_node_values: list=None) -> str:
        if not node_values:
            node_values = ["" for _ in range(number_of_nodes)]

        # Draw nodes
        tikz = ""
        for i in range(number_of_nodes):
            tikz += f"\\node[state" + f", right of={i-1}"*(i != 0) + f"] ({i}) " + "{" + f"{node_values[i]}" + "};\n"

        # Draw stuff in between nodes
        if inter_node_values:
            assert len(inter_node_values) == number_of_nodes - 1

            tikz += "\\path\n"
            for i in range(number_of_nodes-1):
                tikz += f"    ({i}) --node[charstyle] " + "{" + inter_node_values[i] + "} " + f"({i + 1})\n"
            tikz = tikz.rstrip() + ";\n"

        return tikz

    def _wrapWithTikzPicture(self, tikz_body: str) -> str:
        return "\\begin{tikzpicture}[automatastyle, node distance=1.33cm]\n" + indent(1, tikz_body) + "\\end{tikzpicture}\n"

    def visualiseScoreGrid(self, score_grid: ViterbiStepScores, characters: str="",
                           do_numbered_states: bool=False, do_characters: bool=True, do_arc_labels: bool=True, do_alternate_arcs: bool=False) -> str:
        """
        Generates the TikZ code to visualise a Viterbi score grid, using INF as illegal arcs.

        :param score_grid: A characters x steps score grid like you would find in tktkt.models.viterbi.
        """
        # Sanity checks
        do_characters = do_characters and characters

        N,K = score_grid.grid.shape
        assert N > 0
        if do_characters:
            assert len(characters) == N, f"Character dimension has {N} positions even though the string has {len(characters)} characters."

        # Nodes boilerplate
        tikz = self._renderNodes(number_of_nodes=N+1, node_values=list(range(N+1)) if do_numbered_states else None, inter_node_values=characters if do_characters else None)

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
                angle = 20 + i*min(10.0, (90-20)/len(valid_ks))
                tikz += f"    ({n}) edge[bend {arc_direction}={angle}, {label_location}] node[labelstyle] " "{" + f"{round(float(score_grid.get(n,k)),2)}"*do_arc_labels + "}" f" ({n+k+1})\n"
        tikz = tikz.rstrip() + ";\n"

        return self._wrapWithTikzPicture(tikz)

    def visualisePointerGraph(self, graph: SegmentationGraph, node_values: list=None, characters: str="",
                              do_invert_pointers: bool=False, do_alternate_arcs: bool=True, do_arc_labels: bool=True):
        """
        Generates the TikZ code to visualise a backpointer grid.

        :param node_values: Values to put inside the graph nodes.
        :param characters: Values to put between the graph nodes.
        :param do_invert_pointers: Whether to invert the arcs in the graph. Note that this has nothing to do with the
            graph consisting of backpointers versus forepointers, because the whole point of a pointer is that it POINTS
            from somewhere to somewhere, in the correct direction.
        """
        # Sanity checks
        assert len(graph.probabilities) == len(graph.pointers)
        assert all(len(backpointer_sublist) == len(label_sublist)
                   for backpointer_sublist, label_sublist in zip(graph.probabilities, graph.pointers))

        nodes = len(graph.pointers)

        tikz = self._renderNodes(number_of_nodes=nodes, node_values=node_values, inter_node_values=characters)

        # Arcs
        tikz += "\\draw\n"
        for n in range(nodes):
            arc_direction  =  "left" if not do_alternate_arcs or n % 2 == 0 else "right"  # "left" actually means "arc goes over" and "right" means "arc goes under".
            label_location = "above" if not do_alternate_arcs or n % 2 == 0 else "below"
            for i,(n2,l) in enumerate(sorted(zip(graph.pointers[n], graph.probabilities[n]))):
                # Variations on "bend left" include:
                #   - suffixing "left" by "=NUMBERcm" to have the arc deviate from the baseline by a fixed distance
                #   - suffixing "left" by "=NUMBER" to have the arc leave at a unit circle angle in degrees (0 to 90 make sense)
                angle = 20 + i*min(10.0, (90-20)/len(graph.pointers[n]))
                l = round(float(l),2)
                start,end = n,n2  # Pointers are always in the right direction, regardless of whether they're backpointers (start > end) or forepointers (end > start).
                if do_invert_pointers:
                    start,end = end,start
                tikz += f"    ({start}) edge[bend {arc_direction}={angle}, {label_location}" + ", dotted"*(l == 0.0) + "] node[labelstyle] " + "{" + str(l)*do_arc_labels + "}" + f" ({end})\n"
        tikz = tikz.rstrip() + ";\n"

        return self._wrapWithTikzPicture(tikz)
