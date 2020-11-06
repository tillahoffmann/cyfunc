cimport cython
cimport numpy as np
from cyfunc._util cimport fptr, Signature


cdef create_signature(inputs, outputs, fptr func, void* data)


cdef class Cyfunc:
    cdef:
        char* types
        Signature* data
        Signature** data_ptr
        int num_types, num_inputs, num_outputs, num_args
        bytes name, docstring
        object _func

    @staticmethod
    cdef void loop(char **args, np.npy_intp *dimensions, np.npy_intp *steps, void *data)


cdef cython.numeric get_value(char** args, int i, cython.numeric _) nogil
cdef void set_value(char** args, int i, cython.numeric value) nogil
