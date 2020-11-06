cimport cyfunc
cimport numpy as np
import numpy as np
import pytest


cdef void multiply_d(char** args, void* data):
    cdef double x = cyfunc.get_value[double](args, 0, 0)
    cdef double y = cyfunc.get_value[double](args, 1, 0)
    cyfunc.set_value(args, 2, x * y)


signature = cyfunc.create_signature([float, float], [float], multiply_d, NULL)
multiply = cyfunc.register_cyfunc_with_debug("multiply", "multiply two numbers", [signature], False)


def test_create_signature():
    signature = cyfunc.create_signature([float, float], [float], multiply_d, NULL)
    assert signature == {
        'inputs': [np.NPY_DOUBLE, np.NPY_DOUBLE],
        'outputs': [np.NPY_DOUBLE],
        'func': <long>multiply_d,
        'data': 0,
    }
