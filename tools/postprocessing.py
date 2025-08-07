import numpy as np
from scipy.ndimage import label as connected_components

def postprocess_segmentation(pred_mask, keep_largest=True, min_size=100):
    processed = np.zeros_like(pred_mask)

    for cls in np.unique(pred_mask):
        if cls == 0:
            continue  # skip background

        binary = (pred_mask == cls).astype(np.uint8)
        labeled, num = connected_components(binary)

        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        if keep_largest and sizes:
            largest = 1 + np.argmax(sizes)
            processed[labeled == largest] = cls
        else:
            for i in range(1, num + 1):
                if sizes[i-1] >= min_size:
                    processed[labeled == i] = cls

    return processed
