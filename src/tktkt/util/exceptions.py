"""
Errors (i.e. subclasses of Exception) that are so fundamental to tokenisation that
they have their own type rather than just their own message.
"""

class MissingUnkError(Exception):
    """Raised when a type does not have an ID and there is no UNK defined."""
    pass


class EmptyTokenError(Exception):
    """Raised when attempting to claim that the empty string is a token."""
    pass
