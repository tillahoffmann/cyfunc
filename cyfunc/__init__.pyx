from cpython cimport mem
from libc.stdio cimport printf
cimport numpy as np
from ._util cimport fptr

np.import_array()
np.import_ufunc()


cdef struct Signature:  #  Signature struct passed to the ufunc loop function.
    fptr func;  # Pointer to the function to evaluate.
    int num_args;  # Number of arguments.
    void* data;  # Any additional information to pass to the elementwise function.


CYFUNCS = {}  # Container for registered cyfuncs


cdef register_cyfunc_with_debug(name, docstring, signatures, debug):
    """
    Register a cython implementation of a universal function.

    Parameters
    ----------
    name : str
        The name for the ufunc. Specifying a name of `add` or `multiply` enables a special behavior
        for integer-typed reductions when no dtype is given. If the input type is an integer (or
        boolean) data type smaller than the size of the numpy.int_ data type, it will be internally
        upcast to the numpy.int_ (or numpy.uint) data type.
    docstring : str
        Allows passing in a documentation string to be stored with the ufunc. The documentation
        string should not contain the name of the function or the calling signature as that will be
        dynamically determined from the object and available when accessing the __doc__ attribute of
        the ufunc.
    signatures : list[dict]
        Sequence of signatures for the ufunc, one for each dtype combination.

    Returns
    -------
    ufunc : callable
        A universal function.

    Notes
    -----
    See https://numpy.org/doc/stable/reference/c-api/ufunc.html for details.
    """
    cyfunc = Cyfunc(name, docstring, signatures, debug=debug)
    ufunc = cyfunc.ufunc
    CYFUNCS[ufunc] = cyfunc
    return ufunc


cdef register_cyfunc(name, docstring, signatures):
    return register_cyfunc_with_debug(name, docstring, signatures, False)


cdef create_signature(inputs, outputs, fptr func, void* data):
    """
    Create a signature from metadata.

    Parameters
    ----------
    inputs : list
        Sequence of input dtypes.
    outputs : list
        Sequence of output dtypes.
    func : fptr
        Function pointer for elementwise evaluation.
    data : void*
        Additional information to pass to the elementwise function.

    Returns
    -------
    signature : dict
    """
    return {
        'inputs': [np.dtype(x).num for x in inputs],
        'outputs': [np.dtype(x).num for x in outputs],
        'func': <long>func,
        'data': <long>data,
    }


cdef inline cython.numeric get_value(char** args, int i, cython.numeric _) nogil:
    """
    Get the i-th argument.

    Parameters
    ----------
    args : char**
        An array of pointers to the actual data for the input and output arrays. The input arguments
        are given first followed by the output arguments.
    i : int
        Index of the element to get.
    _ :
        Dummy variable to support fused types.

    Notes
    -----
    `args[i]` evaluates to the address of the i-th argument. Thus, (<T*>args[i])[0] is the value of
    the i-th argument of type `T`.

    Examples
    --------
    >>> get_value[double](args, 0)
    """
    return (<cython.numeric*>args[i])[0]


cdef inline void set_value(char** args, int i, cython.numeric value) nogil:
    """
    Set the i-th argument.

    Parameters
    ----------
    args : char**
        An array of pointers to the actual data for the input and output arrays. The input arguments
        are given first followed by the output arguments.
    i : int
        Index of the element to set.
    value : cython.numeric
        Value to set.

    Examples
    --------
    >>> get_value(args, 3.14)
    """
    (<cython.numeric*>args[i])[0] = value


cdef class Cyfunc:
    """
    A container to keep track of the memory associated with a universal function. See
    `register_cyfunc` for details.
    """
    cdef:
        char* types
        Signature* signatures
        Signature** signature_ptr
        int num_types, num_inputs, num_outputs, num_args
        bytes name, docstring
        object _ufunc

    def __init__(self, name, docstring, signatures, *, debug=False):
        # Validate the signatures so we can allocate memory.
        self.num_inputs = self.num_outputs = -1
        for signature in signatures:
            # Make sure the numbers add up for different dtypes
            num_inputs = len(signature['inputs'])
            num_outputs = len(signature['outputs'])
            if self.num_inputs == -1:
                self.num_inputs = num_inputs
            elif self.num_inputs != num_inputs:
                raise ValueError
            if self.num_outputs == -1:
                self.num_outputs = num_outputs
            elif self.num_outputs != num_outputs:
                raise ValueError

        # Allocate the memory.
        self.num_types = len(signatures)
        self.num_args = num_inputs + num_outputs
        self.signatures = <Signature*>mem.PyMem_Malloc(self.num_types * sizeof(Signature))
        self.signature_ptr = <Signature**>mem.PyMem_Malloc(self.num_types * sizeof(Signature*))
        self.types = <char*>mem.PyMem_Malloc(self.num_types * self.num_args * sizeof(char))

        # Fill the allocated memory.
        i = 0
        for j, signature in enumerate(signatures):
            self.signatures[j].num_args = self.num_args
            self.signatures[j].func = <fptr><long>signature['func']
            if 'data' in signature:
                self.signatures[j].data = <void*><long>signature['data']
            else:
                self.signatures[j].data = <void*>0
            self.signature_ptr[j] = &self.signatures[j]

            for input_type in signature['inputs']:
                self.types[i] = input_type
                i += 1
            for output_type in signature['outputs']:
                self.types[i] = output_type
                i += 1

        # Generate the ufunc.
        self.name = name.encode()
        self.docstring = docstring.encode()
        self._ufunc = np.PyUFunc_FromFuncAndData(
            <np.PyUFuncGenericFunction*>(&self.debug_loop if debug else &self.loop),
            <void**>self.signature_ptr,
            self.types,
            self.num_types,
            self.num_inputs,
            self.num_outputs,
            0,  # Identity element
            self.name,
            self.docstring,
            0 # Unused
        )

    def __dealloc__(self):
        mem.PyMem_Free(self.types)
        mem.PyMem_Free(self.signatures)
        mem.PyMem_Free(self.signature_ptr)

    def __str__(self):
        lines = [
            f'num_inputs: {self.num_outputs}',
            f'num_outputs: {self.num_outputs}',
            f'num_args: {self.num_args}',
            f'num_types: {self.num_types}',
            f'types: {self.types}',
            '',
        ]
        for i in range(self.num_types):
            lines.extend([
                f'signature #{i}',
                f'signature address: {<long>self.signature_ptr[i]}',
                f'func: {<long>self.signature_ptr[i].func}',
                f'num_args: {self.signature_ptr[i].num_args}',
                f'data: {<long>self.signature_ptr[i].data}',
            ])
        return '\n'.join(lines)

    @property
    def ufunc(self):
        """
        Return the associated ufunc.
        """
        return self._ufunc

    @staticmethod
    cdef void loop(char **args, np.npy_intp *dimensions, np.npy_intp *steps, void *data):
        """
        General ufunc loop that can accept arbitrary signatures.

        Parameters
        ----------
        args : char **
            An array of pointers to the actual data for the input and output arrays. The input
            arguments are given first followed by the output arguments.
        dimensions : np.npy_intp *
            A pointer to the size of the dimension over which this function is looping.
        steps : np.npy_intp *
            A pointer to the number of bytes to jump to get to the next element in this dimension
            for each of the input and output arguments.
        data : void *
            Arbitrary data (extra arguments, function names, etc.) that can be stored with the
            ufunc and will be passed in when it is called.
        """
        cdef:
            int i, j
            int n = dimensions[0]
            Signature* signature = <Signature*>data;

        # Iterate over the dimensions and apply the function. We use the somewhat unconvential
        # calling order to ensure we don't "over advance" the pointer to the data.
        signature.func(args, signature.data)
        for i in range(n - 1):
            for j in range(signature.num_args):
                args[j] += steps[j]
            signature.func(args, signature.data)

    @staticmethod
    cdef void debug_loop(char **args, np.npy_intp *dimensions, np.npy_intp *steps, void *data):
        """
        Same as `loop` but with extensive print statements.
        """
        cdef:
            int i, j
            np.npy_intp n = dimensions[0]
            Signature* signature = <Signature*>data;

        # Report on what the signature us
        printf("#iterations: %ld\nfunc: %lu\nnum_args: %lu\n===========\n", n, signature.func,
               signature.num_args)

        # Iterate over the dimensions and apply the function
        for i in range(n):
            printf("iteration #%d\nargs: ", i)
            for j in range(signature.num_args):
                printf("%f ", get_value[double](args, j, 0))
            printf("\napplying function...\nargs: ")

            signature.func(args, signature.data)

            for j in range(signature.num_args):
                printf("%f ", get_value[double](args, j, 0))

            if i == n - 1:  # don't advance the pointer on the last loop
                break

            printf("\nsteps: ")
            for j in range(signature.num_args):
                printf("%d ", steps[j])
                args[j] += steps[j]
            printf("\n\n")
