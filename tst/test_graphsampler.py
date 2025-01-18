from tktkt.models.random.graph import ForwardGraphSampler, SegmentationGraph, BackwardGraphSampler


def test_forward():
    sampler = ForwardGraphSampler()
    graph = SegmentationGraph(
        pointers=[[1, 2, 4], [3, 4], [3], [4, 5], [5], []],
        probabilities=[[1 / 3, 1 / 3, 1 / 3], [1 / 2, 1 / 2], [1], [1 / 2, 1 / 2], [1], []]
    )

    assert 6 == sampler.totalPaths(graph)
    for i in range(20):
        path, p = sampler.samplePathAndProb(graph)
        print("\t", path, p)
        assert p == sampler.pathToProb(graph, path)


def test_backward():
    sampler = BackwardGraphSampler()
    graph = SegmentationGraph(
        pointers=[[], [0], [0], [1, 2], [0, 1, 3], [3, 4]],
        probabilities=[[], [1], [1], [1 / 2, 1 / 2], [1 / 3, 1 / 3, 1 / 3], [1 / 2, 1 / 2]]
    )

    assert 6 == sampler.totalPaths(graph)
    for i in range(20):
        path, p = sampler.samplePathAndProb(graph)
        print("\t", path, p)
        assert p == sampler.pathToProb(graph, path)
