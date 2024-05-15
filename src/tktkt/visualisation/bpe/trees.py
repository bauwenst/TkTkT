from typing import List

from ...models.bpe.base import BTE


class BpeTree:

    def __init__(self, token: str, children: List["BpeTree"]=None):
        self.token = token
        self.children = children

    def toForest(self, indent=0):
        s = "[" + self.token
        if self.children is not None:
            s += "\n"
            for child in self.children:
                s += "".join(["\t" + line + "\n" for line in child.toForest(indent + 1).split("\n")])
        s += "]"
        return s


class BpeVisualiser:

    def __init__(self, tokeniser: BTE):
        self.tokeniser = tokeniser

    def tokenise_visualised(self, s: str, intermediates: bool=False):
        """
        Quick-and-dirty implementation of BPE merging, where we keep track of each merge as we go.

        The current segmentation is saved as a list of strings.
        Merges are derived by zipping that list with itself shifted over; hence merges are represented as tuples.
        """
        buffer = list(s)
        current_mergetree_sequence = [BpeTree(c) for c in buffer]
        all_mergetree_sequences = [current_mergetree_sequence.copy()]

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

        return buffer, (current_mergetree_sequence if not intermediates else all_mergetree_sequences)

    def prepareAndTokenise_visualised(self, s: str):
        # Run the visualiser on every pretoken in the string
        tokens = []
        trees  = []
        for pretoken in self.tokeniser.preprocessor.do(s):
            new_tokens, new_trees = self.tokenise_visualised(pretoken)
            tokens.extend(new_tokens)
            trees.extend(new_trees)

        # Convert to LaTeX
        return " ".join(tokens), r"\resizebox{\linewidth}{!}{" + "\n" + BpeVisualiser._treesToLatex(trees) + "}"

    def prepareAndTokenise_visualised_animated(self, s: str):
        """
        Same as above but for Beamer.
        We concatenate the pretokens before tokenising, so that the global order of merges also holds among the
        different pretokens. Let's hope this doesn't create unnatural merges.
        """
        # Run the visualiser on every pretoken in the string
        tokens, many_trees = self.tokenise_visualised("".join(self.tokeniser.preprocessor.do(s)), intermediates=True)

        # Wrap with LaTeX commands and adjust the height
        last_render = BpeVisualiser._treesToLatex(many_trees[-1])

        latex = r"\vphantom{" + last_render + "}%\n"
        for i, trees in enumerate(many_trees):
            latex += r"\only<" + str(i+1) + ">{" + BpeVisualiser._treesToLatex(trees) + "}%\n"

        return " ".join(tokens), r"\resizebox{\linewidth}{!}{" + "\n" + latex + "}"

    @staticmethod
    def _treesToLatex(trees: List[BpeTree]):
        latex = ""
        latex += ("\n" + r"\hskip\forestskip" + "\n").join([
            r"\begin{forest} bpetree" + "\n" +
            tree.toForest() + "\n" +
            r"\end{forest}"
            for tree in trees])
        return latex
