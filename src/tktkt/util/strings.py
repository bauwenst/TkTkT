
def indent(level: int, multiline_string: str, tab: str=" "*4) -> str:
    """
    Prefix each line in the given string by the given tab.
    """
    lines = multiline_string.split("\n")
    lines = [tab*level + line for line in lines]
    return "\n".join(lines)


def alignCharacter(multiline_string: str, character_to_align: str) -> str:
    """
    Add spaces right in front of the first occurrence of the given character for each line in the string, such that
    the character aligns. Useful for aligning equals signs in subsequent assignments.
    """
    lines = multiline_string.split("\n")
    character_locations = [line.find(character_to_align) for line in lines]
    move_character_to = max(character_locations)
    for i, (line, loc) in enumerate(zip(lines, character_locations)):
        if loc < 0:
            continue

        lines[i] = line[:loc] + " "*(move_character_to - loc) + line[loc:]

    return "\n".join(lines)
