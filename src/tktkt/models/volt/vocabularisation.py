"""
Heavily refactored version of the code found at https://github.com/Jingjing-NLP/VOLT
for the Xu e.a. (2021) paper https://aclanthology.org/2021.acl-long.571/.

TODO: In the original code, they don't sort by frequency, but by BPE priority, despite the paper claiming otherwise,
      UNLESS the authors thought that OrderedDict is actually a max-heap and hence their implementation should want to
      keep frequency order instead of merge order.
"""
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict, Counter

import ot

from ...util.iterables import take


def countCharacters(corpus: Path, n_max_lines: int=int(1e7), tokenizer_name: str= 'subword-nmt') -> Counter[str]:
    counts = Counter()
    with open(corpus, "r", encoding="utf-8") as handle:
        for i,line in enumerate(take(n_max_lines,handle)):
            if tokenizer_name == 'subword-nmt':
                line = line.replace("@@ ", "")

            for word in line.split(" "):
                for c in word:
                    if not c.isspace():
                        counts[c] += 1
    return counts


def countCharactersAndFilter(source_corpus: Path, target_corpus: Optional[Path], tokenizer_name='subword-nmt') -> Counter[str]:
    counts = countCharacters(source_corpus, tokenizer_name=tokenizer_name)
    if target_corpus is not None:
        counts += countCharacters(target_corpus, tokenizer_name=tokenizer_name)

    # Filter characters with frequency less than 2.
    return Counter({key: val for key, val in counts.items() if val > 2})


def countTokens(presegmented_corpus: Path, n_max_lines: int=int(1e7), tokenizer_name='subword-nmt'):
    """
    Count tokens in a corpus where they are already separated by spaces.
    """
    counts = Counter()
    with open(presegmented_corpus, "r", encoding="utf-8") as handle:
        for i,line in enumerate(take(n_max_lines,handle)):
            for token in line.split():
                if tokenizer_name == 'subword-nmt' and not token.endswith("@@"):
                    token += "</w>"
                counts[token] += 1

    return counts


def countMergeApplicationsGivenTokenCounts(bpe_merge_file: Path, token_counts: Counter[str], min_number: int=1, tokenizer_name='subword-nmt'):
    """
    Count the amount of times a BPE merge was applied.
    """
    merge_applications: OrderedDict[str,int] = OrderedDict()
    with open(bpe_merge_file, 'r', encoding="utf-8") as handle:
         for line in tqdm(handle):
             if tokenizer_name == 'subword-nmt' and line.startswith("#version"):
                 continue

             merge_string = line.strip()
             parts = merge_string.split(" ")
             merge_result = "".join(parts)
             merge_applications[merge_string] = min_number
             for large_token, freq in token_counts.items():
                 if merge_result in large_token:  # TODO: This is definitely NOT how you measure whether a merge was applied to reach large_token.
                     merge_applications[merge_string] += freq

    return merge_applications


def countMergeApplications(source_corpus: Path, target_corpus: Optional[Path], bpe_merge_file: Path, tokenizer_name='subword-nmt'):
    """
    Get all token candidates associated with their frequencies. Here we take BPE-generated code segmentation as token candidates.
    Arguments:
        source_corpus (str): the source file from machine translation
        target_corpus (str): the target file from machine translation
        bpe_merge_file: the token candidate file. Here we take BPE-generated code segmentation as candidates.
    """
    tokens = countTokens(source_corpus, tokenizer_name=tokenizer_name)
    if target_corpus is not None:
        tokens += countTokens(target_corpus, tokenizer_name=tokenizer_name)
    return countMergeApplicationsGivenTokenCounts(bpe_merge_file, tokens, tokenizer_name=tokenizer_name)


##########################################################################################################


FixedOrderCounts = List[Tuple[str,int]]

def buildDistanceMatrix(chars_with_counts: FixedOrderCounts, merges_with_counts: FixedOrderCounts) -> np.ndarray:
    """
    Initialize distance matrix in optimal transport. if the i-th char in j-th token, their distance is set to be a very small value, otherwize a large value.
    Return a 2-dimension distance matrix.
    """
    matrix = np.zeros((len(chars_with_counts), len(merges_with_counts)))
    rows = len(chars_with_counts)
    cols = len(merges_with_counts)
    for i in range(rows):
        for j in range(cols):
            if chars_with_counts[i][0] in merges_with_counts[j][0]:
                matrix[i][j] = 0.001 * j  # /len(tokens[j][0])#0.00001*tokens[j][1]#-math.log(tokens[j][1]*1.0/total_tokens)*1.0/len(tokens[j][0]) + 0.1/
            else:
                matrix[i][j] = 100  # 0
    return matrix


def getCharacterDistribution(strings_with_counts: FixedOrderCounts, denominator: int, count_individual_characters: bool=False):
    """
    Compute how many characters are represented by each item in the given sequence.
        - For characters, this is just their frequencies.
        - For tokens, this is their frequencies multiplied by their length: for example, give a token 'cat' with
          frequency 500, it requires 500 'c', 500 'a', and 500 't'. Therefore, it requires 1500 characters in total.

    The result is not a probability distribution due to this reweighting.

    :param count_individual_characters: whether to take token length into account or treat everything like a character.
    """
    return [
        (len(token) if count_individual_characters else 1) * freq/denominator + 1e-4
        for token,freq in strings_with_counts
    ]


def pruneMerges(candidate_merges: Counter[str], chars: Counter[str],
                p_matrix: np.ndarray, threshold: float=0.0001) -> List[str]:
    """
    Conserve merges based on the optimal-transport matrix P.

    :param p_matrix: The P matrix as generated by Sinkhorn optimal transport. Should be indexed on characters and tokens
                     in such a way that the highest-frequency tokens and characters are the lowest indices.
    :param threshold: The filter ratio from the optimal transportation matrix to the real vocabulary. Here we set a
                      small value. Higher threshold means that more tokens are removed from the token candidates.
    """
    candidate_merges = candidate_merges.most_common()
    chars            = chars.most_common()
    vocab_size = len(candidate_merges)
    merges_to_chars_to_score = dict()  # Maps a merge string to each character and each character to some kind of number. The number seems purely for logging reasons; what this dictionary actually measures is whether there exists any character that has a high enough score when interacting with a given merge.
    for j in tqdm(range(len(p_matrix[0]))):
        merge_string, merge_count = candidate_merges[j]
        if merge_string.strip() == "":
            continue

        merges_to_chars_to_score[merge_string] = dict()
        for i in range(len(p_matrix)):
            char = chars[i][0]
            merge_versus_char_score = p_matrix[i][j]*vocab_size
            if p_matrix[i][j] != 0 and merge_versus_char_score > threshold*merge_count:
                merges_to_chars_to_score[merge_string][char] = merge_versus_char_score  # * len(tokens[j][0])

    final_merges = []
    deleted_actions = dict()
    types_necessary_for_merges = dict()  # Once again, this is a dictionary for logging purposes only. The key set is what you want. The values just explain for which merge the given type is necessary.
    for merge_string in merges_to_chars_to_score.keys():
        if len(merges_to_chars_to_score[merge_string]) > 0:
            left, right = merge_string.split(" ")
            types_necessary_for_merges[left]  = merge_string
            types_necessary_for_merges[right] = merge_string
        else:
            pass
            # print(merge_string, merges_to_chars_to_score[merge_string])
            # deleted_actions[merge_string.replace(" ", "")] = merge_string

    for merge_string in merges_to_chars_to_score.keys():
        merge_result = merge_string.replace(" ", "")
        if merge_result in types_necessary_for_merges or len(merges_to_chars_to_score[merge_string]) > 0:
            final_merges.append(merge_string)
        else:
            pass
            # print(merge_string, merges_to_chars_to_score[merge_string])

    return final_merges


def searchBestSize(candidate_merges: Counter[str], chars: Counter[str],
                   max_number: int=30000, interval: int=1000, n_max_iters: int=300) -> int:
    scores = {}
    previous_entropy = 0

    sorted_chars: FixedOrderCounts = chars.most_common()
    total_chars = chars.total()

    sorted_merges_all: FixedOrderCounts = candidate_merges.most_common()
    for Vsize in range(interval, max_number, interval):  # iteration_numbers:
        sorted_merges_current = sorted_merges_all[:Vsize]
        total_merges = sum(map(lambda merge_with_freq: merge_with_freq[1], sorted_merges_current))

        d_matrix = buildDistanceMatrix(sorted_chars, sorted_merges_current)
        a = getCharacterDistribution(sorted_chars, total_chars)
        b = getCharacterDistribution(sorted_merges_current, total_merges, True)

        epsilon = 0.1  # entropy parameter
        alpha = 1.0  # Unbalanced KL relaxation parameter
        current_entropy, _ = ot.sinkhorn(a, b, d_matrix, 1.0, method='sinkhorn', numItermax=n_max_iters, epsilon0=1e-6)
        if Vsize == interval:
            # print("finish reading", iter_number, Gs, (Gs-current_entropy)/2)
            previous_entropy = current_entropy
        else:
            # print("finish running", iter_number, current_entropy, current_entropy-previous_entropy)
            scores[Vsize] = current_entropy - previous_entropy
            previous_entropy = current_entropy

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Best size:", sorted_scores[0][0])
    print("One optional solution is that you can use this size to generated vocabulary in subword-nmt or sentencepiece")
    return sorted_scores[0][0]


def solveTransportMatrix(candidate_merges: Counter[str], chars: Counter[str],
                         optimal_size: int, n_max_iters: int=300) -> np.ndarray:
    sorted_chars: FixedOrderCounts = chars.most_common()
    total_chars = chars.total()

    sorted_tokens: FixedOrderCounts = candidate_merges.most_common(optimal_size)
    total_tokens = sum(map(lambda token_freq: token_freq[1], sorted_tokens))

    d_matrix = buildDistanceMatrix(sorted_chars, sorted_tokens)
    a = getCharacterDistribution(sorted_chars, total_chars)
    b = getCharacterDistribution(sorted_tokens, total_tokens, True)

    epsilon = 0.1  # entropy parameter
    alpha = 1.0  # Unbalanced KL relaxation parameter
    _, P = ot.sinkhorn(a, b, d_matrix, 1.0, method='sinkhorn', numItermax=n_max_iters, epsilon0=1e-6)
    return P


def volt(source_file: Path, target_file: Optional[Path], token_candidate_file: Path, tokenizer_name: str,
         max_number: int=10_000, interval: int=1000, num_iter_max: int=500, threshold: float=0.00001):
    merge_counts = countMergeApplications(source_file, target_file, token_candidate_file, tokenizer_name=tokenizer_name)  # get token candidates and their frequencies
    char_counts  = countCharactersAndFilter(source_file, target_file, tokenizer_name=tokenizer_name)  # get chars and their frequencies
    optimal_size = searchBestSize(merge_counts, char_counts, max_number, interval, num_iter_max)  # generate the best ot size
    P = solveTransportMatrix(merge_counts, char_counts, optimal_size, num_iter_max)  # generate the optimal matrix based on the ot size
    return pruneMerges(merge_counts, char_counts, P, threshold)  # generate the vocabulary based on the optimal matrix
