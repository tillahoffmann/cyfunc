# Function pointer declaration for elementwise callable
ctypedef void (*fptr)(char**, void*)


# Signature struct passed to the ufunc loop function
cdef struct Signature:
    fptr func;  # Pointer to the function to evaluate
    int num_args;  # Number of arguments
    void* data;  # Any additional information to pass to the elementwise function
