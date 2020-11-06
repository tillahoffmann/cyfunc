import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, language_level=3)
import _test_cyfunc

# Inject thin wrappers of test functions into the global namespace
test_functions = {key: lambda: getattr(_test_cyfunc, key)() for key in dir(_test_cyfunc)
                  if key.startswith('test')}
globals().update(test_functions)
