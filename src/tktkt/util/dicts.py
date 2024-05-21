
def invertdict(d: dict, noninjective_ok=True) -> dict:
    """
    Keys become values, values become keys.
    Values must be hashable in that case.
    """
    d_inv = {v: k for k,v in d.items()}

    # Uh oh!
    if not noninjective_ok and len(d) != len(d_inv):
        values_with_multiple_keys = dict()

        # Redo the construction and keep track of keys with duplicate values.
        d_inv = dict()
        for k,v in d.items():
            if v in d_inv:  # Already in d_inv, so another key had this value.
                if v not in values_with_multiple_keys:
                    values_with_multiple_keys[v] = [d_inv[v]]
                values_with_multiple_keys[v].append(k)
            else:
                d_inv[v] = k

        raise ValueError(f"Dictionary could not be inverted because it wasn't injective. The following values were associated with more than one key: {values_with_multiple_keys}")

    return d_inv


def insertKeyAlias(d: dict, existing_key: str, alias_key: str) -> dict:
    """
    In-place, but still returns the given dictionary.
    """
    d[alias_key] = d[existing_key]
    return d