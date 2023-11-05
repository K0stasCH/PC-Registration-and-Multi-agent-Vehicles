import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def mask2image(mask, palette):
    seg_img = Image.fromarray(mask).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    return seg_img

# if __name__=='__main__':
#     maskPath = 'D:\\Dataset-Thesis\\temp\V2X Sim Mini\\V2X-Sim-2.0-mini\\sweeps\\SEG_FRONT_id_1\\scene_5_000006.npz'

#     classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
#                  'traffic light', 'traffic sign', 'vegetation', 'terrain',
#                  'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
#                  'motorcycle', 'bicycle')
#     pallete=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
#                  [190, 153, 153], [153, 153, 153], [250, 170,
#                                                     30], [220, 220, 0],
#                  [107, 142, 35], [152, 251, 152], [70, 130, 180],
#                  [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
#                  [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

#     with np.load(maskPath) as data:
#         mask = data['arr_0']

#     # create a patch (proxy artist) for every color
#     patches = [mpatches.Patch(color=np.array(pallete[i])/255.,
#                             label=classes[i]) for i in range(8)]
#     # put those patched as legend-handles into the legend
#     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#             fontsize='large')

#     img = mask2image(mask, pallete)
#     plt.imshow(img)
#     plt.show()


#     print(1)