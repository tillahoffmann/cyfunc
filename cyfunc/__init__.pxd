cimport cython
from cyfunc._util cimport fptr


cdef create_signature(inputs, outputs, fptr func, void* data)
cdef register_cyfunc(name, docstring, signatures)
cdef register_cyfunc_with_debug(name, docstring, signatures, debug)
cdef cython.numeric get_value(char** args, int i, cython.numeric _) nogil
cdef void set_value(char** args, int i, cython.numeric value) nogil
