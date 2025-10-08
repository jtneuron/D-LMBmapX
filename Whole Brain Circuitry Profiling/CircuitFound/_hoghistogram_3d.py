import numpy as np

def cell_hog_3d(magnitude, orientation, orientation_start, orientation_end,
                cell_columns, cell_rows, cell_depth,
                column_index, row_index, depth_index,
                size_columns, size_rows, size_depth,
                range_rows_start, range_rows_stop,
                range_columns_start, range_columns_stop,
                range_depth_start, range_depth_stop):
    """Calculation of the cell's HOG value for 3D data.

    Parameters
    ----------
    magnitude : ndarray
        The gradient magnitudes of the voxels.
    orientation : ndarray
        Lookup table for orientations.
    orientation_start : float
        Orientation range start.
    orientation_end : float
        Orientation range end.
    cell_columns : int
        Voxels per cell (columns).
    cell_rows : int
        Voxels per cell (rows).
    cell_depth : int
        Voxels per cell (depth).
    column_index : int
        Block column index.
    row_index : int
        Block row index.
    depth_index : int
        Block depth index.
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    size_depth : int
        Number of depth slices.
    range_rows_start : int
        Start row of cell.
    range_rows_stop : int
        Stop row of cell.
    range_columns_start : int
        Start column of cell.
    range_columns_stop : int
        Stop column of cell.
    range_depth_start : int
        Start depth of cell.
    range_depth_stop : int
        Stop depth of cell.

    Returns
    -------
    total : float
        The total HOG value.
    """
    total = 0.0

    for cell_depth in range(range_depth_start, range_depth_stop):
        depth_index_cell = depth_index + cell_depth
        if (depth_index_cell < 0 or depth_index_cell >= size_depth):
            continue

        for cell_row in range(range_rows_start, range_rows_stop):
            row_index_cell = row_index + cell_row
            if (row_index_cell < 0 or row_index_cell >= size_rows):
                continue

            for cell_column in range(range_columns_start, range_columns_stop):
                column_index_cell = column_index + cell_column
                if (column_index_cell < 0 or column_index_cell >= size_columns
                        or orientation[depth_index_cell, row_index_cell, column_index_cell] >= orientation_start
                        or orientation[depth_index_cell, row_index_cell, column_index_cell] < orientation_end):
                    continue

                total += magnitude[depth_index_cell, row_index_cell, column_index_cell]

    return total / (cell_depth * cell_rows * cell_columns)


def hog_histograms_3d(gradient_columns, gradient_rows, gradient_depth,
                      cell_columns, cell_rows, cell_depth,
                      size_columns, size_rows, size_depth,
                      number_of_cells_columns, number_of_cells_rows, number_of_cells_depth,
                      number_of_orientations, orientation_histogram):
    """Extract Histogram of Oriented Gradients (HOG) for a given 3D image.

    Parameters
    ----------
    gradient_columns : ndarray
        First order image gradients (columns).
    gradient_rows : ndarray
        First order image gradients (rows).
    gradient_depth : ndarray
        First order image gradients (depth).
    cell_columns : int
        Voxels per cell (columns).
    cell_rows : int
        Voxels per cell (rows).
    cell_depth : int
        Voxels per cell (depth).
    size_columns : int
        Number of columns.
    size_rows : int
        Number of rows.
    size_depth : int
        Number of depth slices.
    number_of_cells_columns : int
        Number of cells (columns).
    number_of_cells_rows : int
        Number of cells (rows).
    number_of_cells_depth : int
        Number of cells (depth).
    number_of_orientations : int
        Number of orientation bins.
    orientation_histogram : ndarray
        The histogram array which is modified in place.
    """

    magnitude = np.sqrt(gradient_columns**2 + gradient_rows**2 + gradient_depth**2)
    orientation = np.rad2deg(np.arctan2(np.sqrt(gradient_columns**2 + gradient_rows**2), gradient_depth)) % 180

    r_0 = cell_rows // 2
    c_0 = cell_columns // 2
    d_0 = cell_depth // 2
    cc = cell_columns * number_of_cells_columns
    cr = cell_rows * number_of_cells_rows
    cd = cell_depth * number_of_cells_depth
    range_rows_stop = (cell_rows + 1) // 2
    range_rows_start = -(cell_rows // 2)
    range_columns_stop = (cell_columns + 1) // 2
    range_columns_start = -(cell_columns // 2)
    range_depth_stop = (cell_depth + 1) // 2
    range_depth_start = -(cell_depth // 2)
    number_of_orientations_per_180 = 180. / number_of_orientations

    for i in range(number_of_orientations):
        # Isolate orientations in this range
        orientation_start = number_of_orientations_per_180 * (i + 1)
        orientation_end = number_of_orientations_per_180 * i
        r = r_0
        d = d_0
        r_i = 0
        d_i = 0

        while d < cd:
            r_i = 0
            r = r_0

            while r < cr:
                c_i = 0
                c = c_0

                while c < cc:
                    orientation_histogram[d_i, r_i, c_i, i] = cell_hog_3d(
                        magnitude,
                        orientation,
                        orientation_start,
                        orientation_end,
                        cell_columns,
                        cell_rows,
                        cell_depth,
                        c,
                        r,
                        d,
                        size_columns,
                        size_rows,
                        size_depth,
                        range_rows_start,
                        range_rows_stop,
                        range_columns_start,
                        range_columns_stop,
                        range_depth_start,
                        range_depth_stop,
                    )
                    c_i += 1
                    c += cell_columns

                r_i += 1
                r += cell_rows

            d_i += 1
            d += cell_depth
