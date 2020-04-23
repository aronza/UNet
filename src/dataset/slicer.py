import logging

def build_slices(input_shape, patch_shape=(64, 128, 128), stride_shape=(32, 128, 128)):
    """Iterates over a given n-dim dataset patch-by-patch with a given stride
                    and builds an array of slice positions.

                    Returns:
                        list of slices, i.e.
                        [(slice, slice, slice, slice), ...] if len(shape) == 4
                        [(slice, slice, slice), ...] if len(shape) == 3
                    """
    slices = []
    if len(input_shape) == 4:
        in_channels, i_z, i_y, i_x = input_shape
    else:
        i_z, i_y, i_x = input_shape

    k_z, k_y, k_x = patch_shape
    s_z, s_y, s_x = stride_shape
    z_steps = gen_indices(i_z, k_z, s_z)
    for z in z_steps:
        y_steps = gen_indices(i_y, k_y, s_y)
        for y in y_steps:
            x_steps = gen_indices(i_x, k_x, s_x)
            for x in x_steps:
                slice_idx = (
                    slice(z, z + k_z),
                    slice(y, y + k_y),
                    slice(x, x + k_x)
                )
                if len(input_shape) == 4:
                    slice_idx = (slice(0, in_channels),) + slice_idx
                slices.append(slice_idx)
    logging.info(f'Patch shape {patch_shape}')
    
    return slices


def gen_indices(i, k, s):
    assert i >= k, 'Sample size has to be bigger than the patch size'
    for j in range(0, i - k + 1, s):
        yield j
    if j + k < i:
        yield i - k
