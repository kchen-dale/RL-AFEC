import numpy as np
from video_quality import load_model


def get_video_quality(loss_pattern):

    NN_model = load_model()
    # Calling the model directly is much faster than predict() for a single sample.
    vmaf = NN_model(loss_pattern, training=False).numpy()
    # VMAF is defined on [0, 100]; the regressor can slightly overshoot either end.
    vmaf = np.clip(vmaf, 0, 100)

    return vmaf

