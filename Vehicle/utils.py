import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def mask2image(mask, palette):
    # seg_img = Image.fromarray(mask.).convert('P')
    seg_img = Image.fromarray((mask).astype(np.float64)).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    return seg_img

def plotTrajectory(vehicles:list):
    """
    plot the ground truth trajectories of the vehicles in the same plot
    """
    from .Vehicle import Vehicle
    if vehicles is not None:
        if not all(isinstance(v, Vehicle) for v in vehicles):
            raise TypeError("All elements in othervehicles must be instances of the Vehicle class.")

    for v in vehicles:
        trajectory = np.asarray(v.egoTranslation_Stream)
        transparency = np.linspace(0, 1, num=len(trajectory))
        plt.scatter(trajectory[:, 0], trajectory[:, 1], marker='o', 
                        linestyle='-', label=f'Vehicle {v.vehicle_id}', alpha=transparency)

    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.show()
    return