"""Image warping using per-pixel flow vectors."""

import numpy as np


def interpolate_bilinear(
    grid,
    query_points,
    indexing: str = "ij",
    xp = np,
):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).
    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    grid = xp.array(grid)
    query_points = xp.array(query_points)

    # grid shape checks
    assert grid.ndim == 4
    assert grid.shape[1] >=2
    assert grid.shape[2] >= 2

    # query_points shape checks
    assert query_points.ndim == 3
    assert query_points.shape[-1] == 2

    batch_size, height, width, channels = grid.shape
    num_queries = query_points.shape[1]

    query_type = query_points.dtype
    grid_type = grid.dtype

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]

    for i, dim in enumerate(index_order):
        queries = query_points[...,dim]
        # queries = unstacked_query_points[dim]
        size_in_indexing_dimension = grid.shape[i + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = size_in_indexing_dimension - 2
        # min_floor = tf.constant(0.0, dtype=query_type)
        floor = xp.minimum(
           xp.maximum(0., xp.floor(queries)), max_floor
        ).astype(query_type)
        int_floor = floor.astype(xp.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).astype(grid_type)
        alpha = xp.minimum(xp.maximum(0., alpha), 1.)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = xp.expand_dims(alpha, 2)
        alphas.append(alpha)

        flattened_grid = xp.reshape(grid, [batch_size * height * width, channels])
        batch_offsets = xp.reshape(
            xp.arange(batch_size) * height * width, [batch_size, 1]
        )

    def gather(y_coords, x_coords):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = xp.take_along_axis(flattened_grid, linear_coordinates, axis=0)
        return xp.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])

    # now, do the actual interpolation
    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def dense_image_warp(image, flow, xp=np):
    """Image warping using per-pixel flow vectors.
    Apply a non-linear warp to the image, where the warp is specified by a
    dense flow field of offset vectors that define the correspondences of
    pixel values in the output image back to locations in the source image.
    Specifically, the pixel value at `output[b, j, i, c]` is
    `images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c]`.
    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    `(b, j - flow[b, j, i, 0], i - flow[b, j, i, 1])`. For locations outside
    of the image, we use the nearest pixel values at the image boundary.
    NOTE: The definition of the flow field above is different from that
    of optical flow. This function expects the negative forward flow from
    output image to source image. Given two images `I_1` and `I_2` and the
    optical flow `F_12` from `I_1` to `I_2`, the image `I_1` can be
    reconstructed by `I_1_rec = dense_image_warp(I_2, -F_12)`.
    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).
      Note that image and flow can be of type `tf.half`, `tf.float32`, or
      `tf.float64`, and do not necessarily have to be the same type.
    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.
    Raises:
      ValueError: if `height < 2` or `width < 2` or the inputs have the wrong
        number of dimensions.
    """
    image = xp.array(image)
    flow = xp.array(flow)
    batch_size, height, width, channels = image.shape

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = xp.meshgrid(xp.arange(width), xp.arange(height))
    #grid_x = grid_x - tf.math.reduce_mean(grid_x)
    #grid_y = grid_y - tf.math.reduce_mean(grid_y)
    stacked_grid = xp.stack([grid_y, grid_x], axis=2).astype(flow.dtype)
    batched_grid = xp.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = xp.reshape(
        query_points_on_grid, [batch_size, height * width, 2]
    )

    # Compute values at the query points, then reshape the result back to the image grid
    interpolated = interpolate_bilinear(image, query_points_flattened, xp=xp)
    interpolated = xp.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated