from keras.optimizers import Adam

import numpy as np
import os.path
import matplotlib.pyplot as plt
import csv

import Kepler_model
import preprocess
import environment
import get_single_keplerid

# gui imports
import tkinter as tk
global category
global revperiod
global dist
global radius
global widgets


ASTRONOMICAL_UNIT = 1.495978707*(10**11)
EARTH_RADIUS = 6378


class tce_struct:
    kepid = 0
    tce_period = 0.0
    tce_sma = 0.0
    tce_prad = 0.0
    tce_duration = 0.0
    tce_time0bk = 0.0


def predict_by_kepler_tce(tce):
    X1, X2 = [], []

    model = Kepler_model.build_Kepler_CNN()

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if environment.NB_CLASSES >= 10:
        metrics.append('top_k_categorical_accuracy')

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    # optimizer = 'adadelta'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=metrics)

    # Load the existing trained model
    trained_model_filename = os.path.join(
        environment.KEPLER_TRAINED_MODEL_FOLDER, 'kepler-model.h5')

    if os.path.isfile(trained_model_filename):
        model.load_weights(trained_model_filename)

    # =========================================================================
    # Get the light-curve files

    # First, look for the KEPLER_DATA_FOLDER, see if we already have that Kepler ID downloaded
    kepid_formatted = "{0:09d}".format(int(tce.kepid))  # Pad with zeros.

    download_folder = os.path.join(
        environment.KEPLER_DATA_FOLDER, kepid_formatted[0:4], kepid_formatted)

    from_unclassified_folder = False

    if not os.path.exists(download_folder):
        # We didn't have the light-curve files downloaded for this Kepler ID
        # Go to check if we have it downloaded to the KEPLER_UNCLASSIFIED_DATA_FOLDER
        download_folder = os.path.join(environment.DATA_FOR_PREDICTION_FOLDER,
                                       kepid_formatted[0:4], kepid_formatted)

        print("Target not in KEPLER_DATA_FOLDER. Look into the DATA_FOR_PREDICTION_FOLDER.")
        from_unclassified_folder = True

        if not os.path.exists(download_folder):
            # We don't have it downloaded to KEPLER_UNCLASSIFIED_DATA_FOLDER yet.
            # Do the download first
            print("Need to download to the DATA_FOR_PREDICTION_FOLDER first.")
            get_single_keplerid.download_one_kepler_id_files(tce.kepid)

    # =========================================================================
    # Get the global view and local view from the light-curve files

    if not from_unclassified_folder:
        time, flux = preprocess.read_and_process_light_curve(
            tce.kepid, environment.KEPLER_DATA_FOLDER, 0.75)
    else:
        time, flux = preprocess.read_and_process_light_curve(
            tce.kepid, environment.DATA_FOR_PREDICTION_FOLDER, 0.75)

    time, flux = preprocess.phase_fold_and_sort_light_curve(
        time, flux, tce.tce_period, tce.tce_time0bk)

    global_view = preprocess.global_view(time, flux, tce.tce_period)
    local_view = preprocess.local_view(
        time, flux, tce.tce_period, tce.tce_duration)

    # Change the dimension to fit for the model input shape
    global_view = np.reshape(global_view, (2001, 1))
    local_view = np.reshape(local_view, (201, 1))

    # =========================================================================
    # Save the global view and local view to a picture
    fig, axes = plt.subplots(1, 2, figsize=(10 * 2, 5), squeeze=False)
    axes[0][0].plot(global_view, ".")
    axes[0][0].set_title("Global view")
    axes[0][0].set_xlabel("Bucketized Time (days)")
    axes[0][0].set_ylabel("Normalized Flux")

    axes[0][1].plot(local_view, ".")
    axes[0][1].set_title("Local view")
    axes[0][1].set_xlabel("Bucketized Time (days)")
    axes[0][1].set_ylabel("Normalized Flux")

    fig.tight_layout()

    file_name = '{0}_period={1}_time0bk={2}_duration={3}.png'\
        .format(tce.kepid, tce.tce_period, tce.tce_time0bk, tce.tce_duration*24)
    file_name = os.path.join(environment.PREDICT_OUTPUT_FOLDER, file_name)

    if os.path.isfile(file_name):
        os.remove(file_name)

    fig.savefig(file_name, bbox_inches="tight")

    # =========================================================================
    # Do the prediction
    X1.append(global_view)

    result_X = np.array(X1)

    predict_result = model.predict(result_X, batch_size=1, verbose=0)

    predict_result_text = ""
    predict_result_index = 0

    if predict_result[0][0] > 0.5:
        predict_result_text = "PC (planet candidate)"
        predict_result_index = 0

    if predict_result[0][1] > 0.5:
        predict_result_text = "Not a planet candidate "
        predict_result_index = 1

    '''
    if predict_result[0][2] > 0.5:
        predict_result_text = "NTP (non-transiting phenomenon)"
        predict_result_index = 2
    '''
    if predict_result[0][0] > 0.5:
        print("\nKepler ID    = {:9d}".format(tce.kepid))
        print("Revolutionary Period = {} days".format(tce.tce_period))
        print("Maximum Dsitance From Star= {} km".format(tce.tce_sma))
        print("Planetary Radius = {} km".format(tce.tce_prad))
        revperiod = tce.tce_period
        dist = tce.tce_sma
        radius = tce.tce_prad
        category = "{0:.00%} possibility is a {1:s}".format(
            predict_result[0][predict_result_index], predict_result_text)
    if predict_result[0][1] > 0.5:
        tce.tce_period = "Unavailable"
        tce.tce_sma = "Unavailable"
        tce.tce_prad = "Unavailable"
        print("\nKepler ID    = {:9d}".format(tce.kepid))
        print("Revolutionary Period = Unavailable")
        print("Maximum Dsitance From Star= Unavailable")
        print("Planetary Radius = Unavailable")
        revperiod = tce.tce_period
        dist = tce.tce_sma
        radius = tce.tce_prad
        category = "{0:.00%} possibility is a {1:s}".format(
            predict_result[0][predict_result_index], predict_result_text)

    # Print the duration in hours value
    # print("tce_duration = {}".format(tce.tce_duration))

    print("\n==> Predicted result = {0}".format(predict_result))

    # We are using the two-class category: ['0_PC', '1_NON_PC'] instead
    # print("==> Available labels: {}".format(environment.ALLOWED_LABELS))
    print("==> Available labels: {}".format(['0_PC', '1_NON_PC']))
    print("==> {0:.00%} possibility is a {1:s}".format(
        predict_result[0][predict_result_index], predict_result_text))

    # fig.show()
    return revperiod, dist, radius, category


# gui window
def close_window():
    global entry
    global ID
    ID = entry.get()

    tce = tce_struct()
    tce.kepid = int(ID)
    with open(environment.KEPLER_CSV_FILE) as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:

            keplerid = row["kepid"]
            keplerid = int(keplerid)
            if(tce.kepid == keplerid):

                tc_dur = row["tce_duration"]
                tc_0bk = row["tce_time0bk"]
                tc_period = row["tce_period"]
                tc_prad = row["tce_prad"]
                tc_sma = row["tce_sma"]
                tce.tce_sma = float(tc_sma)*ASTRONOMICAL_UNIT
                tce.tce_prad = float(tc_prad)*EARTH_RADIUS
                tce.tce_period = float(tc_period)
                tce.tce_time0bk = float(tc_0bk)
                tce.tce_duration = float(tc_dur)
    tce.tce_duration /= 24
    revperiod, dist, radius, category = predict_by_kepler_tce(tce)

    for widget in frame.winfo_children():
        widget.destroy()
    cat = tk.Text(frame, height=1, width=40)
    cat.insert(tk.INSERT, category)
    cat.configure(state='disabled')
    cat.pack(fill=tk.X, side='top')
    rev = tk.Label(frame, text='Revolutionary Period:')
    rev.pack()
    entry1 = tk.Text(frame, height=1, width=10)
    entry1.insert(tk.INSERT, revperiod)
    entry1.configure(state='disabled')
    entry1.pack(fill=tk.X)
    distance = tk.Label(frame, text='Distance from star:')
    distance.pack()
    entry2 = tk.Text(frame, height=1, width=10)
    entry2.insert(tk.INSERT, dist)
    entry2.configure(state='disabled')
    entry2.pack(fill=tk.X)
    rad = tk.Label(frame, text='Radius of planet:')
    rad.pack()
    entry3 = tk.Text(frame, height=1, width=10)
    entry3.insert(tk.INSERT, radius)
    entry3.configure(state='disabled')
    entry3.pack(fill=tk.X)


window = tk.Tk()
window.minsize(400, 400)
label = tk.Label(window, text="Kepler ID")
label.pack()
entry = tk.Entry(window, bd=5)
entry.pack(fill=tk.X, side='top')
B = tk.Button(window, text="Enter", command=close_window)
B.pack(fill=tk.X, side='top')
frame = tk.Frame(window)
frame.pack(fill=tk.X, side='top')
window.mainloop()

#tce = tce_struct()

# tce.kepid = 11442793
# tce.tce_period = 331.603
# tce.tce_time0bk = 140.48
# tce.tce_duration = 14.49
"""
    tce.kepid = 11442793
    with open(environment.KEPLER_CSV_FILE) as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:

            keplerid = row["kepid"]
            keplerid = int(keplerid)
            if(tce.kepid == keplerid):

                tc_dur = row["tce_duration"]
                tc_0bk = row["tce_time0bk"]
                tc_period = row["tce_period"]
                tc_prad = row["tce_prad"]
                tc_sma = row["tce_sma"]
                tce.tce_sma = float(tc_sma)*ASTRONOMICAL_UNIT
                tce.tce_prad = float(tc_prad)*EARTH_RADIUS
                tce.tce_period = float(tc_period)
                tce.tce_time0bk = float(tc_0bk)
                tce.tce_duration = float(tc_dur)
"""
'''
    tce.kepid = 757450
    tce.tce_period = 8.88492
    tce.tce_time0bk = 134.452
    tce.tce_duration = 2.078  # in hours
    '''
"""
    # Convert duration to days
    tce.tce_duration /= 24

    rp, dis, rad, categ = predict_by_kepler_tce(tce)
"""
