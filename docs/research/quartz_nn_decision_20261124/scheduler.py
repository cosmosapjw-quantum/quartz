"""Initial red scaffold: wrong policy chooses most uncertain, ignores sign."""
def signed_order(scores, ids):
    return sorted(range(len(scores)),key=lambda i:ids[i])
