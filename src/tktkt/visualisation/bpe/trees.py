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
    """
    Tracker for historically applied merges. Also renders the actual visualisation.
    """

    def __init__(self, token: str, children: List["BpeTree"]=None):
        self.token: str = token
        self.children: List[BpeTree] = children or []

    def toForest(self, weighting_function: VisualWeightingFunction=None):
        raw_weights = None if weighting_function is None else self._getRawWeights(weighting_function, depth=0)
        return self._toForest(depth=0, weighting_function=weighting_function, all_tree_weights=raw_weights)

    def _getRawWeights(self, weighting_function: VisualWeightingFunction, depth: int) -> List[float]:
        """
        For this tree, compute the list of results gained from applying the weighting function recursively (before any normalisation).
        """
        results = [weighting_function.getTokenWeight(token=self.token, depth=depth, is_leaf=not self.children)]
        for child in self.children:
            results.extend(child._getRawWeights(weighting_function, depth+1))
        return results

    def _toForest(self, depth: int, weighting_function: VisualWeightingFunction=None, all_tree_weights: List[float]=None):
        if weighting_function is None:
            content = self.token
        else:
            content = r"$\underset{\color{black!35}" + \
                      f"{weighting_function.getTokenWeightGivenAll(all_tree_weights, self.token, depth=depth, is_leaf=not self.children):.3f}" + \
                      r"\vphantom{{}^0}}{\smash{\text{" + self.token + r"}}}$"  # The vphantom is to create enough space between the token and the subscript. The smash is to prevent more space in case the token has a descender (e.g. the letter p).

        s = "[" + content
        if self.children:
            s += "\n"
            for child in self.children:
                s += "".join(["\t" + line + "\n" for line in child._toForest(depth+1, weighting_function, all_tree_weights).split("\n")])
        s += "]"
        return s


MergeTrace = List[BpeTree]

class BpeVisualiser:

    def __init__(self, tokeniser: BTE, weighting_function: VisualWeightingFunction=None):
        self.tokeniser = tokeniser
        self.weighter = weighting_function

    def tokenise_visualised(self, pretoken: str, intermediates: bool=False) -> Tuple[List[str], Union[MergeTrace, List[MergeTrace]]]:
        return self.applyMerges_visualised(self.tokeniser.boundary_marker.intoCharacters(pretoken), intermediates=intermediates)

    def applyMerges_visualised(self, characters: Iterable[str], intermediates: bool=False) -> Tuple[List[str], Union[MergeTrace, List[MergeTrace]]]:
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

        if intermediates:
            return buffer, all_mergetree_sequences
        else:
            return buffer, current_mergetree_sequence

    def prepareAndTokenise_visualised(self, s: str) -> Tuple[str, str]:
        # Run the visualiser on every pretoken in the string
        tokens = []
        trees  = []
        for pretoken in self.tokeniser.preprocessor.do(s):
            new_tokens, new_trees = self.tokenise_visualised(pretoken)
            tokens.extend(new_tokens)
            trees.extend(new_trees)

        # Convert to LaTeX
        return " ".join(tokens), r"\resizebox{\linewidth}{!}{" + "\n" + self._treesToLatex(trees) + "}"

    def prepareAndTokenise_visualised_animated(self, s: str) -> Tuple[str, str]:
        """
        Same as above but for Beamer.
        We concatenate the pretokens before tokenising, so that the global order of merges also holds among the
        different pretokens. Let's hope this doesn't create unnatural merges.
        """
        # Run the visualiser on every pretoken in the string
        tokens, many_trees = self.tokenise_visualised("".join(self.tokeniser.preprocessor.do(s)), intermediates=True)

        # Wrap with LaTeX commands and adjust the height
        last_render = self._treesToLatex(many_trees[-1])

        latex = r"\vphantom{" + last_render + "}%\n"
        for i, trees in enumerate(many_trees):
            latex += r"\only<" + str(i+1) + ">{" + self._treesToLatex(trees) + "}%\n"

        return " ".join(tokens), r"\resizebox{\linewidth}{!}{" + "\n" + latex + "}"

    def _treesToLatex(self, trees: MergeTrace):
        latex = ""
        latex += ("\n" + r"\hskip\forestskip" + "\n").join([
            r"\begin{forest} bpetree" + ("-weighted" if self.weighter is not None else "") + "\n" +
            tree.toForest(weighting_function=self.weighter) + "\n" +
            r"\end{forest}"
            for tree in trees])
        return latex

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
}

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
        return p if forked_edges else p.replace("forked edges", "")
