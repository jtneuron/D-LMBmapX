import matplotlib.pyplot as plt
import numpy as np


def preview_image(image_array, is_show=False, normalize_by="volume", cmap=None, figsize=(18, 6), threshold=None,
                  **kwargs):
    """
    Display three orthogonal slices of the given 3D image.

    image_array is assumed to be of shape (H,W,D)

    If a number is provided for threshold, then pixels for which the value
    is below the threshold will be shown in red
    """
    if normalize_by == "slice":
        vmin = None
        vmax = None
    elif normalize_by == "volume":
        vmin = 0
        vmax = image_array.max().item()
    else:
        raise (ValueError(
            f"Invalid value '{normalize_by}' given for normalize_by"))

    # half-way slices
    if len(image_array.shape) == 3:
        x, y, z = np.array(image_array.shape) // 2
    elif len(image_array.shape) == 4:
        x, y, z = np.array(image_array.shape[:-1]) // 2
    else:
        raise Exception(f"image_array shape length is {len(image_array.shape)}, but it should be 3 or 4.")
    imgs = (image_array[x, :, :], image_array[:, y, :], image_array[:, :, z])

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, im in zip(axs, imgs):
        ax.axis('off')
        ax.imshow(im, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

        # threshold will be useful when displaying jacobian determinant images;
        # we will want to clearly see where the jacobian determinant is negative
        if threshold is not None:
            red = np.zeros(im.shape + (4,))  # RGBA array
            red[im <= threshold] = [1, 0, 0, 1]
            ax.imshow(red, origin='lower', **kwargs)
    if is_show:
        plt.show()
    return fig


from matplotlib import colors


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
