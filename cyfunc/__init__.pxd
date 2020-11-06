cimport cython
from cyfunc._util cimport fptr


cdef create_signature(inputs, outputs, fptr func, void* data)
cdef register_cyfunc(name, docstring, signatures)
cdef cython.numeric get_value(char** args, int i, cython.numeric _) nogil
cdef void set_value(char** args, int i, cython.numeric value) nogil
