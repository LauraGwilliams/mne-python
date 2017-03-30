# Authors: Laura Gwilliams <laura.gwilliams@nyu.edu>
#          Jean-Remi King <jeanremi.king@gmail.com>
#          Alex Barachant <alexandre.barachant@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)


# Steps:
# 1) preprocess raw into epochs band-passed at different frequencies and
#    cropped with X many cycles
# 2) fit classifier with cross validation
# 3) predict
# 4) score

import numpy as np

from mne.filter import filter_data
from mne.decoding import CSP


class TimeFrequencyCSP(object):

    def __init__(self, tmin, tmax, freqs, estimator, n_cycles=7., sfreq=None,
                 scorer=None, n_jobs=1):

        # add test that the number of cycles is not too many for the low freq

        # add input testing (e.g. mne.decoding.csp.CSP)
        self.freqs = freqs
        self.estimator = estimator
        self.n_cycles = n_cycles
        self.sfreq = sfreq
        self.scorer = scorer
        self.n_jobs = n_jobs

        # Assemble list of frequency range tuples
        self._freq_ranges = zip(self.freqs[:-1], freqs[1:])

        # Infer window spacing from the max freq and n cycles to avoid gaps
        self._window_spacing = (n_cycles / np.max(freqs) / 2.)
        self._centered_w_times = np.arange(tmin+self._window_spacing,
                                           tmax-self._window_spacing,
                                           self._window_spacing)[1:]  # noqa
        self._n_windows = len(self._centered_w_times)

    def _transform(self, epochs, y, w_tmin, w_tmax, method='', pos=[]):
        """
        Assumes data for one frequency band for one time window and fits CSP.
        """

        from sklearn import clone

        # Crop data into time-window of interest
        Xt = epochs.copy().crop(w_tmin, w_tmax).get_data()

        # call fit or predict depending on calling function
        if method == 'fit':
            self.estimator = clone(self.estimator)
            self._estimators[pos] = (self.estimator.fit(Xt, y))
            return self

        elif method == 'predict':
            self.y_pred = self._estimators[pos].predict(Xt)

        elif method == 'score':
            self.y_pred = self._estimators[pos].predict(Xt)
            print self.y_pred
            return self._estimators[pos].score(Xt, y)

    def fit(self, epochs, y):

        n_freq = len(self._freq_ranges)
        n_window = len(self._centered_w_times)
        self._scores = np.zeros([n_freq, n_window])
        self._estimators = np.empty([n_freq, n_window], dtype=object)

        # TODO: change the work flow such that the bandpass is only done once
        # per frequency. this means moving the freq loop and the window loop
        # to the fit/predict/score function. then the cropping is done in
        # some generic "transform-like" function, which calls fit/predict/
        # score depending on the method passed. If it is fit, we need to Save
        # the estimators for each window and freq. if it is predict we need to
        # return y_pred to compare to y_true at the scoring stage.

        # all of the cross validation is done outside of the functions

        for freq_ii, (fmin, fmax) in enumerate(self._freq_ranges):
            # Infer window size based on the frequency being used
            w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # filter the data at the desired frequency
            epochs_bp = epochs.copy()
            epochs_bp._data = filter_data(epochs_bp.get_data(),
                                            self.sfreq, fmin, fmax)

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(self._centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # extract data for this time window
                self._transform(epochs_bp, y, w_tmin, w_tmax, method='fit',
                                pos=(freq_ii, t))
        pass

    def predict(self, epochs):

        for freq_ii, (fmin, fmax) in enumerate(self._freq_ranges):
            # Infer window size based on the frequency being used
            w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # filter the data at the desired frequency
            epochs_bp = epochs.copy()
            epochs_bp._data = filter_data(epochs_bp.get_data(),
                                            self.sfreq, fmin, fmax)

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(self._centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # extract data for this time window
                self._transform(epochs_bp, None, w_tmin, w_tmax,
                                method='predict', pos=(freq_ii, t))
        pass

    def score(self, epochs, y):

        for freq_ii, (fmin, fmax) in enumerate(self._freq_ranges):
            # Infer window size based on the frequency being used
            w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

            # filter the data at the desired frequency
            epochs_bp = epochs.copy()
            epochs_bp._data = filter_data(epochs_bp.get_data(),
                                            self.sfreq, fmin, fmax)

            # Roll covariance, csp and lda over time
            for t, w_time in enumerate(self._centered_w_times):

                # Center the min and max of the window
                w_tmin = w_time - w_size / 2.
                w_tmax = w_time + w_size / 2.

                # extract data for this time window
                s = self._transform(epochs_bp, y, w_tmin, w_tmax,
                                    method='score', pos=(freq_ii, t))
                self._scores[freq_ii, t] = s

        return self._scores





# Load data

# Set parameters and read data
from mne import Epochs, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

# add imports in object's method, remove noqa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline  # noqa
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

event_id = dict(hands=2, feet=3)  # motor imagery: hands vs feet
subject = 1
runs = [6, 10, 14]
raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raw_files)

# Extract information from the raw file
sfreq = raw.info['sfreq']
events = find_events(raw, shortest_event=0, stim_channel='STI 014')
raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# Classification & Time-frequency parameters
estimator = make_pipeline(CSP(n_components=4, reg=None, log=True),
                          LinearDiscriminantAnalysis())
n_splits = 5  # how many folds to use for cross-validation
cv = KFold(n_splits=n_splits, shuffle=True)

# Classification & Time-frequency parameters
tmin, tmax = -.200, 2.000
n_cycles = 10.  # how many complete cycles: used to define window size
min_freq = 5.
max_freq = 25.
n_freqs = 8  # how many frequency bins to use

epochs = Epochs(raw, events, event_id, tmin * 10, tmax * 10,
                    proj=False, baseline=None, add_eeg_ref=False, preload=True)
y = epochs.events[:, 2] - 2
sfreq = epochs.info['sfreq']

freqs = np.linspace(min_freq, max_freq, n_freqs)
tf = TimeFrequencyCSP(tmin, tmax, freqs, estimator,
                      sfreq=sfreq)

scores = []
for train, test in cv.split(epochs.copy().get_data()):
    tf.fit(epochs[train], y[train])
    scores.append(tf.score(epochs[test], y[test]))
ave_scores = np.mean(np.array(scores), axis=0)

# Set up time frequency object
import matplotlib.pyplot as plt
from mne.time_frequency import AverageTFR
from mne import create_info

av_tfr = AverageTFR(create_info(['freq'], sfreq), ave_scores[np.newaxis, :],
                    tf._centered_w_times, freqs[1:], 1)

chance = np.mean(y)  # set chance level to white in the plot
av_tfr.plot([0], vmin=chance, title="Time-Frequency Decoding Scores",
            cmap=plt.cm.Reds)
