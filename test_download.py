from __future__ import absolute_import

import kepler_io
import matplotlib.pyplot as plt
import numpy as np


KEPLER_DATA_DIR = "D:/K2_Light_Curves_Dataset/"
KEPLER_ID = 11442793  # Kepler-90.


# Read the light curve.
file_names = kepler_io.kepler_filenames(KEPLER_DATA_DIR, KEPLER_ID)
assert file_names, "Failed to find .fits files in {}".format(KEPLER_DATA_DIR)
all_time, all_flux = kepler_io.read_kepler_light_curve(file_names)
print("Read light curve with {} segments".format(len(all_time)))


# Plot the fourth segment.
plt.plot(all_time[3], all_flux[3], ".")
plt.show()

# Plot all light curve segments. We first divide by the median flux in each
# segment, because the segments are on different scales.
for f in all_flux:
    f /= np.median(f)
plt.plot(np.concatenate(all_time), np.concatenate(all_flux), ".")
plt.show()
