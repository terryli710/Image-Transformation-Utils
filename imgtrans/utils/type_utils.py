

# check types of variables

import collections


def is_array_like(x):
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))

