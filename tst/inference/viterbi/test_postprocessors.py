from tktkt.models.viterbi.objectives_postprocessors import DiscretiseScores
from tktkt.models.viterbi.objectives_unguided import RandomScoreGenerator


def test_discretisation():
    word = "abcdef"
    k = 6

    generator = DiscretiseScores(RandomScoreGenerator(), minimum_score=0, maximum_score=1, discretisation_levels=3)
    grid = generator.nested_generator.generateGrid(word, k)
    print(grid)
    generator.augmentScores(grid, word, k)
    print(grid)


if __name__ == "__main__":
    test_discretisation()
