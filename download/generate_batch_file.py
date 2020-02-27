from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import stat
import sys

wget_cmd = ("wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off "
             "-R 'index*' -A _llc.fits")
url_b="http://archive.stsci.edu/pub/kepler/lightcurves"
dr24csv_file="D:/Final_Year_Project/dr24tce.csv"
download_directory="D:/K2_Light_Curves_Dataset"

kepler_ids = set()
with open(dr24csv_file) as f:
  reader = csv.DictReader(row for row in f if not row.startswith("#"))
  for row in reader:
    kepler_ids.add(row["kepid"])

num_kepids = len(kepler_ids)

with open("download/download.bat","w") as f:

  f.write("ECHO 'Downloading {} Kepler targets to {}'\n".format(num_kepids, download_directory))
  for i, kepid in enumerate(kepler_ids):
    print("Next")
    if i and not i % 10:
      f.write("echo 'Downloaded {}/{}'\n".format(i, num_kepids))
      kepid = "{0:09d}".format(int(kepid))
      subdir = "{}/{}".format(kepid[0:4], kepid)
      download_dir = os.path.join(download_directory, subdir)
      url = "{}/{}/".format(url_b, subdir)
      f.write("{} -P {} {}\n".format(wget_cmd, download_dir, url))
  f.write("echo 'Finished downloading {} Kepler targets to {}'\n".format(num_kepids, download_directory))


print("Completed")
