from tktkt.util.dicts import ChainedCounter


def test_chained():
    c = ChainedCounter(5)
    c["a"] += 2
    c["a"] += 4
    c["a"] += 1
    c["b"] += 2

    c.update(["c", "a", "b"])
    print(c._counters)

    # Note that [key] and .get(key) don't use the same implementation.
    print(c["a"])
    print(c.get("a"))

    c["c"] += 12
    print(c._counters)
    print(c)
    print(c.total())
