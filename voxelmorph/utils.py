import itertools
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
    
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    
    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """


    volshape = loc_shift.shape[:-1].as_list()
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return interpn(vol, loc, interp_method=interp_method)



def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'

    Returns:
        new interpolated volume of the same size as the entries in loc

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """
    
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    # since loc can be a list, nb_dims has to be based on vol.
    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')
    
    volshape = vol.shape.as_list()

    # interpolate
    if interp_method == 'linear':
        loc0 = tf.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.get_shape().as_list()]
        clipped_loc = [tf.clip_by_value(loc[...,d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0] # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector 
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        
        for c in cube_pts:
            
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            wt = K.expand_dims(wt, -1)
            
            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val
        
    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')

        # clip values
        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        roundloc = [tf.clip_by_value(roundloc[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx) 

    return interp_vol



def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    TODO: 
        allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """

    volshape = volshape.as_list()
    
    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)) :
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  
    mesh = [tf.cast(f, 'float32') for f in mesh]
    
    if shift_center:
        mesh = [mesh[f] - (volshape[f]-1)/2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)



def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))
    k = np.cumprod(siz[::-1])
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]
    return ndx


def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod

def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """
    
    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)

def meshgrid(*args, **kwargs):
    """
    # 这里可以用tensorflow自己的函数替代，它说会有性能上的问题，没有测试过
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921
    
    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """

    indexing = kwargs.pop("indexing", "xy")
    name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype

    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow. 
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # TODO(nolivia): improve performance with a broadcast  
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):       
        output[i] = tf.tile(output[i], tf.stack([*sz[:i], 1, *sz[(i+1):]]))
    return output
    
def flatten(v):
    """
    flatten Tensor v
    
    Parameters:
        v: Tensor to be flattened
    
    Returns:
        flat Tensor
    """

    return tf.reshape(v, [-1])