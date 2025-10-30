from typing import List, Union, Tuple, Iterable
from abc import ABC, abstractmethod

import numpy as np

from ...models.bpe.base import BTE


class NormalisationFunction(ABC):

    @abstractmethod
    def normalise(self, value: float, all_values: List[float]):
        pass


class IdentityNormalisation(NormalisationFunction):

    def normalise(self, value: float, all_values: List[float]):
        return value


class LinearNormalisation(NormalisationFunction):

    def normalise(self, value: float, all_values: List[float]):
        return value / sum(all_values)


class SoftmaxNormalisation(NormalisationFunction):

    def __init__(self, temperature: float=1.0, scale_beforehand: bool=False):
        self.tau = temperature
        self.do_scale = scale_beforehand

    def normalise(self, value: float, all_values: List[float]):
        all_values = np.array(all_values)
        if self.do_scale:
            denominator = np.sum(all_values)
            value      = value / denominator
            all_values = all_values / denominator

        shift = np.max(all_values)  # (implementation detail for better softmaxes)
        exp_value      =        np.exp(1/self.tau * (value - shift))
        sum_exp_values = np.sum(np.exp(1/self.tau * (all_values - shift)))
        return exp_value / sum_exp_values


class VisualWeightingFunction(ABC):
    """
    Provides weights for tokens at a given depth in the tree,
    which can be visualised alongside a tree node.

    (Cannot pinpoint exact tokens because there is no way to do that.)
    """

    def __init__(self, normaliser: NormalisationFunction=IdentityNormalisation()):
        self.normaliser = normaliser

    @abstractmethod
    def getTokenWeight(self, token: str, depth: int, is_leaf: bool) -> float:
        """
        Get the weight for the given token given only the other arguments as context.
        """
        pass

    def getTokenWeightGivenAll(self, all_tree_weights: List[float], token: str, depth: int, is_leaf: bool) -> float:
        """
        Get the weight given all weights in whatever tree this token is visualised inside of.
        """
        return self.normaliser.normalise(value=self.getTokenWeight(token, depth, is_leaf), all_values=all_tree_weights)


class ExponentialDepthWeighting(VisualWeightingFunction):
    """
    Example weighting scheme. A node at depth d (starting at 0) has weight 2^{-2*d-1}, unless it is a leaf, which gets 2^{-2*d}.
    For example, see the tree:

            a 0.5           --> sum to 1.0
          /       \         --> sum to 0.5
        b 0.25    c 0.125   --> sum to 0.25
                 /      \   --> sum to 0.125
              d 0.0625  e 0.0625
    """

    def getTokenWeight(self, token: str, depth: int, is_leaf: bool) -> float:
        return 2**(-2*depth - 1 + is_leaf)


class BpeTree:
    """Tracker for historically applied merges."""

    def __init__(self, token: str, children: List["BpeTree"]=None):
        self.token: str = token
        self.children: List[BpeTree] = children or []
        self.weight = 1


class BpeTreeWeightAssigner(ABC):

    @abstractmethod
    def assignWeights(self, tree: BpeTree):
        pass


class AssignWeightsFromFlattenedTree(BpeTreeWeightAssigner):
    """
    Generates weights for the BFS-ordered flat tree.
    """

    def assignWeights(self, tree: BpeTree):
        weights = self.getFlattenedWeights(tree)

        frontier = [tree]
        while frontier:
            new_frontier = []
            for tree in frontier:
                tree.weight = weights.pop(0)
                new_frontier.extend(tree.children)
            frontier = new_frontier

    @abstractmethod
    def getFlattenedWeights(self, tree: BpeTree) -> List[float]:
        pass


class AssignWeightsUsingTokenFunction(BpeTreeWeightAssigner):

    def __init__(self, token_weighting_function: VisualWeightingFunction):
        self.weighter = token_weighting_function

    def getFlattenedWeights(self, tree: BpeTree) -> List[float]:
        weights = []

        # Step 1: Collect unnormalised weights.
        frontier = [tree]
        depth = 0
        while frontier:
            new_frontier = []
            for tree in frontier:
                weights.append(self.weighter.getTokenWeight(token=self.token, depth=depth, is_leaf=not tree.children))
                new_frontier.extend(tree.children)
            frontier = new_frontier
            depth += 1

        # Step 2: Exact same process, but now knowing the full list of weights beforehand.
        normalised_weights = []

        frontier = [tree]
        depth = 0
        while frontier:
            new_frontier = []
            for tree in frontier:
                normalised_weights.append(self.weighter.getTokenWeightGivenAll(weights, token=tree.token, depth=depth, is_leaf=not tree.children))
                new_frontier.extend(tree.children)
            frontier = new_frontier
            depth += 1

        return normalised_weights


MergeTrace = List[BpeTree]

class BpeVisualiser:

    def __init__(self, tokeniser: BTE):
        self.tokeniser = tokeniser

    def _applyMerges_visualised(self, characters: Iterable[str]) -> Tuple[List[str], List[MergeTrace]]:
        """
        Quick-and-dirty implementation of BPE merging, where we keep track of each merge as we go.

        The current segmentation is saved as a list of strings.
        Merges are derived by zipping that list with itself shifted over; hence merges are represented as tuples.
        """
        buffer = list(characters)
        current_mergetree_sequence: MergeTrace = [BpeTree(c) for c in buffer]
        all_mergetree_sequences: List[MergeTrace] = [current_mergetree_sequence.copy()]

        merges_to_ranks = {tuple(m.parts): m.priority for m in self.tokeniser.merge_graph.merges}
        merges = set(merges_to_ranks.keys())

        hypothetical_merges = set(zip(buffer[:-1], buffer[1:]))
        actual_merges = hypothetical_merges & merges
        while actual_merges:
            priority_merge = sorted(actual_merges, key=lambda m: merges_to_ranks[m])[0]
            new_token = "".join(priority_merge)

            length_to_iterate = len(buffer) - 1
            i = 0
            while i < length_to_iterate:
                if buffer[i] == priority_merge[0] and buffer[i+1] == priority_merge[1]:
                    buffer[i:i+2] = [new_token]  # Python allows this :o
                    current_mergetree_sequence[i:i+2] = [BpeTree(new_token, [current_mergetree_sequence[i], current_mergetree_sequence[i+1]])]
                    length_to_iterate -= 1
                i += 1
            all_mergetree_sequences.append(current_mergetree_sequence.copy())

            hypothetical_merges = set(zip(buffer[:-1], buffer[1:]))
            actual_merges = hypothetical_merges & merges

        return buffer, all_mergetree_sequences

    def tokenise_visualised(self, pretoken: str) -> Tuple[List[str], MergeTrace]:
        tokens, traces = self._applyMerges_visualised(self.tokeniser._boundary_marker.atomise(pretoken))
        return tokens, traces[-1]

    def prepareAndTokenise_visualised(self, s: str) -> Tuple[List[str], str]:
        # Run the visualiser on every pretoken in the string
        tokens = []
        trees  = []
        for pretoken in self.tokeniser.preprocessor.do(s):
            new_tokens, new_trees = self.tokenise_visualised(pretoken)
            tokens.extend(new_tokens)
            trees.extend(new_trees)

        # Convert to LaTeX
        return tokens, r"\resizebox{\linewidth}{!}{" + "\n" + self._treesToLatex(trees) + "}"

    def tokenise_visualised_animated(self, pretoken: str) -> Tuple[List[str], List[MergeTrace]]:
        return self._applyMerges_visualised(self.tokeniser._boundary_marker.atomise(pretoken))

    def prepareAndTokenise_visualised_animated(self, s: str) -> Tuple[str, str]:
        """
        Same as above but for Beamer.
        We concatenate the pretokens before tokenising, so that the global order of merges also holds among the
        different pretokens. Let's hope this doesn't create unnatural merges.
        """
        # Run the visualiser on every pretoken in the string
        tokens, many_trees = self.tokenise_visualised_animated("".join(self.tokeniser.preprocessor.do(s)))

        # Wrap with LaTeX commands and adjust the height
        last_render = self._treesToLatex(many_trees[-1])

        latex = r"\vphantom{" + last_render + "}%\n"
        for i, trees in enumerate(many_trees):
            latex += r"\only<" + str(i+1) + ">{" + self._treesToLatex(trees) + "}%\n"

        return " ".join(tokens), r"\resizebox{\linewidth}{!}{" + "\n" + latex + "}"

    def _treesToLatex(self, trees: MergeTrace) -> str:
        latex = ""
        latex += ("\n" + r"\hskip\forestskip" + "\n").join([
            r"\begin{forest} " + self._forestStyle() + "\n" +
            self._treeToLatex(tree) + "\n" +
            r"\end{forest}"
            for tree in trees])
        return latex

    def _treeToLatex(self, tree: BpeTree) -> str:
        s = "[" + self._formatTreeNode(tree)
        if tree.children:
            s += "\n"
            for child in tree.children:
                child_string = self._treeToLatex(child)
                s += "".join(["\t" + line + "\n" for line in child_string.split("\n")])
        s += "]"
        return s

    def _formatTreeNode(self, tree: BpeTree) -> str:
        return tree.token

    def _forestStyle(self) -> str:
        return "bpetree"

    @staticmethod
    def getLatexPreamble(forked_edges: bool=False):
        p = r"""
\newlength{\forestskip}
\setlength{\forestskip}{1 mm}

\forestset{
    bpetree/.style = {
        % Configure tree so it can be stacked with other trees: https://tex.stackexchange.com/a/687344/203081
        delay={where content={}{shape=coordinate}{}},
        where n children=0{tier=word, baseline, font=\scshape}{},
        for tree={
            align=center,
            base=bottom,
            text height = 2ex,
            text depth  = 0.5ex,
            inner ysep = 0pt,
            outer ysep = 3pt,
            inner xsep = 1pt,
            s sep = \forestskip{} + 1mm % <------- 
        },
        % Drop nodes as low as possible: https://tex.stackexchange.com/a/686577/203081
        for tree children-first={
          if n children=0{
            tier=0,
          }{
            tier/.max={1+tier}{children},
          },
        },
        forked edges
    }
}"""
        return p if forked_edges else p.replace("forked edges", "")


class BpeVisualiserWeighted(BpeVisualiser):

    def __init__(self, tokeniser: BTE, weight_assigner: BpeTreeWeightAssigner):
        super().__init__(tokeniser)
        self.weight_assigner = weight_assigner

    def _formatTreeNode(self, tree: BpeTree) -> str:
        return r"$\underset{\color{black!35}" + \
               f"{tree.weight:.3f}" + \
               r"\vphantom{{}^0}}{\smash{\text{" + tree.token + r"}}}$"  # The vphantom is to create enough space between the token and the subscript. The smash is to prevent more space in case the token has a descender (e.g. the letter p).

    def _treesToLatex(self, trees: MergeTrace) -> str:
        for tree in trees:
            self.weight_assigner.assignWeights(tree)
        return super()._treesToLatex(trees)

    def _forestStyle(self) -> str:
        return "bpetree-weighted"

    @staticmethod
    def getLatexPreamble(forked_edges: bool = False):
        p = r"""
\newlength{\forestskip}
\setlength{\forestskip}{1 mm}

\forestset{
    bpetree-weighted/.style = {
        % Configure tree so it can be stacked with other trees: https://tex.stackexchange.com/a/687344/203081
        delay={where content={}{shape=coordinate}{}},
        where n children=0{tier=word, baseline, font=\scshape}{},
        for tree={
            parent anchor=south,  % When enabled, stick the child edges together into one point.
            % anchor=north, 
            child anchor=north,
            %
            text height = 1.25ex,  % Whitespace on the top.
            text depth  = 1.75ex,  % Whitespace on the bottom.
            % inner ysep = 5ex,  % This is kind of a combination of height and depth. outer ysep is the same.
            s sep = \forestskip{} + 1mm
        },
        % Drop nodes as low as possible: https://tex.stackexchange.com/a/686577/203081
        for tree children-first={
          if n children=0{
            tier=0,
          }{
            tier/.max={1+tier}{children},
          },
        },
        forked edges
    }
}"""