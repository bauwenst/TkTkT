from tst.preamble import *

from tktkt.preparation.perturbers import *

def test_pops():
    example = "This is a specific test to see how many deletions happen."

    sampler = FixedUniformSampler(1, 2)

    pop = Pop(p=1.0, sampler=sampler)
    print(pop.perturb(example))

    sampler = GeometricSampler(1, 10, 0.9, start_at_max=True)
    pop = Pop(p=1.0, sampler=sampler)
    print(pop.perturb(example))
