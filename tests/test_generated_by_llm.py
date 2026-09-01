import pytest

def calculate_sum(a, b):
    return a + b

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),                     # positive integers
        (0, 0, 0),                     # zeros
        (-5, 5, 0),                    # negative and positive cancel out
        (-3, -7, -10),                 # both negative
        (1.5, 2.5, 4.0),               # floats
        (1e10, 2e10, 3e10),            # large numbers
        (0, -0, 0),                    # negative zero (int)
        (float('inf'), 1, float('inf')),  # infinity
        (float('-inf'), -1, float('-inf')), # negative infinity
        (float('nan'), 0, float('nan')),   # NaN propagation
    ],
)
def test_calculate_sum_normal(a, b, expected):
    result = calculate_sum(a, b)
    if isinstance(expected, float) and (expected != expected):  # NaN check
        assert result != result
    else:
        assert result == expected


@pytest.mark.parametrize(
    "a, b, exc_type",
    [
        ("1", 2, TypeError),          # string and int
        (None, 5, TypeError),         # None and int
        ([1, 2], 3, TypeError),       # list and int
        ({"a": 1}, 2, TypeError),     # dict and int
    ],
)
def test_calculate_sum_type_errors(a, b, exc_type):
    with pytest.raises(exc_type):
        calculate_sum(a, b)
