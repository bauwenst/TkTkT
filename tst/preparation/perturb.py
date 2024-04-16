from tktkt.preparation.perturbers import *

def test_pops():
    example = "This is a specific test to see how many deletions happen."
    pop = RandomPop(p=1.0, min_n=1, max_n=2)
    print(pop.perturb(example))

    pop = GeometricPop(p=1.0, min_n=1, max_n=10, probability_to_stop=0.9, start_at_max=True)
    print(pop.perturb(example))
