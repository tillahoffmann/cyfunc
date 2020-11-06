import numpy as np
import pytest
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, language_level=3)
import _test_cyfunc  # noqa: E402
from _test_cyfunc import multiply  # noqa: E402

# Inject thin wrappers of test functions into the global namespace
test_functions = {key: lambda: getattr(_test_cyfunc, key)() for key in dir(_test_cyfunc)
                  if key.startswith('test')}
globals().update(test_functions)


def test_evaluation():
    x = np.arange(3)
    y = np.pi
    actual = multiply(x, y)
    np.testing.assert_allclose(actual, x * y)


def test_evaluation_out():
    x = np.arange(3)
    y = np.pi
    actual = np.empty(3)
    multiply(x, y, out=actual)
    np.testing.assert_allclose(actual, x * y)


@pytest.mark.parametrize('shape1, shape2', [
    [(3,), ()],
    [(15,), ()],
    [(15, 3), (3,)],
    [(117, 5, 1), (5, 13)],
])
def test_evaluation_where(shape1, shape2):
    x = 1.0 + np.arange(np.prod(shape1)).reshape(shape1)
    y = 2 + np.arange(np.prod(shape2)).reshape(shape2)
    desired = -np.ones_like(x + y)
    actual = -np.ones_like(x + y)
    where = np.random.uniform(size=shape1) < 0.5

    np.multiply(x, y, out=desired, where=where)
    multiply(x, y, out=actual, where=where)

    np.testing.assert_allclose(actual, desired)
