import theano.tensor as T

from lasagne.layers.base import MergeLayer



class SpatialElemwiseMergeLayer(MergeLayer):
    """
    This layer performs an elementwise merge of its input layers.
    It requires the number of units in the second input layer to have
    the same as the one of channels in the first input layer.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal
    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.
    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`
    See Also
    --------
    ElemwiseSumLayer : Shortcut for sum layer.
    """

    def __init__(self, img, coeff, merge_function, **kwargs):
        super(SpatialElemwiseMergeLayer, self).__init__((img, coeff), **kwargs)
        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        print input_shapes
        if input_shapes[0][1]!=input_shapes[1][1]:
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        img = inputs[0]
        coeff = inputs[1].dimshuffle(0, 1, 'x', 'x')
        output = self.merge_function(img, coeff)
        return output

