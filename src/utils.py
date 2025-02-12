def get_key_by_value(dictionary, target_value):
    """
    Retrieves the key associated with the given value in a dictionary.

    Args:
        dictionary (dict): The dictionary to search through.
        target_value (any): The value to find the corresponding key for.

    Returns:
        any: The key associated with the target value, or None if the value is not found.
    """
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None