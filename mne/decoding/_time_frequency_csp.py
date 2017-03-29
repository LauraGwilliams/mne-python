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

# from .base import _set_cv

from mne.filter import band_pass_filter
from mne.decoding import CSP


# add imports in object's method, remove noqa
from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline


class TimeFrequencyCSP(object):

    def __init__(self, tmin, tmax, min_freq, max_freq, freq_bins, cv,
                 estimator, n_cycles=7., sfreq=None, scorer=None, n_jobs=1):

        # replace min_freq, max_freq and freqs_bin into freqs
        # remove cv
        # add input testing (e.g. mne.decoding.csp.CSP)
        self.tmin = tmin
        self.tmax = tmax
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_bins = freq_bins
        self.cv = cv
        self.estimator = estimator
        self.n_cycles = n_cycles
        self.sfreq = sfreq
        self.scorer = scorer
        self.n_jobs = n_jobs

        # Assemble list of frequency range tuples
        self.freqs = np.linspace(min_freq, max_freq, freq_bins)
        self.freq_ranges = zip(self.freqs[:-1], self.freqs[1:])

        # if constructed parameters they must be hidden using a _ before

        # Infer window spacing from the max freq and n cycles to avoid gaps
        self.window_spacing = (n_cycles / np.max(self.freqs) / 2.)
        self.centered_w_times = np.arange(tmin, tmax, self.window_spacing)[1:]
        self.n_windows = len(self.centered_w_times)

    def _transform(self, epochs, fmin, fmax, w_tmin, w_tmax):
        """
        Assumes data for one frequency band for one time window and fits CSP.
        """

        # filter the data at the desired frequency
        epochs._data = band_pass_filter(epochs.get_data(), self.sfreq, fmin,
                                        fmax)

        # Crop data into time-window of interest
        Xt = epochs.copy().crop(w_tmin, w_tmax).get_data()

        return Xt

    def _loop_tf(self, epochs, y=None, method=None):

        flen = len(self.freq_ranges)  # change to n_freqs
        tlen = len(self.centered_w_times)  # n_windows
        n_splits = self.cv.n_splits

        self._csps = []
        self._scores = np.zeros([flen, tlen, n_splits])

        for split_idx, (train, test) in enumerate(self.cv_splits):

            for freq_ii, (fmin, fmax) in enumerate(self.freq_ranges):
                # Infer window size based on the frequency being used
                w_size = n_cycles / ((fmax + fmin) / 2.)  # in seconds

                # Roll covariance, csp and lda over time
                for t, w_time in enumerate(self.centered_w_times):

                    # Center the min and max of the window
                    w_tmin = w_time - w_size / 2.
                    w_tmax = w_time + w_size / 2.

                    # extract data for this frequency band and this time window
                    Xt = self._transform(epochs, fmin, fmax, w_tmin, w_tmax)

                    # call fit or predict depending on calling function
                    if method == 'fit':
                        self._fit_slice(Xt[train], y[train])
                    elif method == 'predict':
                        self._predict_slice(Xt[test])
                    elif method == 'score':
                        self._predict_slice(Xt[test])
                        self._score_slice(Xt[test], y[test],
                                         (freq_ii, t, split_idx))

        return self

    def _fit_slice(self, Xt, y):

        _csp = self.estimator.fit(Xt, y)
        self._csps.append(_csp)

        return self

    def fit(self, epochs, y):

        # TODO: change the work flow such that the bandpass is only done once
        # per frequency. this means moving the freq loop and the window loop
        # to the fit/predict/score function. then the cropping is done in
        # some generic "transform-like" function, which calls fit/predict/
        # score depending on the method passed. If it is fit, we need to Save
        # the estimators for each window and freq. if it is predict we need to
        # return y_pred to compare to y_true at the scoring stage.

        self.cv_splits = cv.split(epochs.get_data())

        return self._loop_tf(epochs, y, method='fit')

    def _predict_slice(self, Xt):

        _csp = self.estimator.predict(Xt)
        self._csps.append(_csp)

        return self

    def predict(self, epochs):

        return self._loop_tf(epochs, method='predict')

    def _score_slice(self, Xt, y, pos=[]):

        scores = self.estimator.score(Xt, y)
        self._scores[pos] = (scores)

        return scores

    def score(self, epochs, y):

        scores = self._loop_tf(epochs, y, method='score')
        scores = np.mean(scores, axis=-1)  # remove because cv outside

        return scores


# Load data

# Set parameters and read data
from mne import Epochs, find_events
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

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

tmin, tmax = -.500, 1.000
n_cycles = 4.  # how many complete cycles: used to define window size
min_freq = 5.
max_freq = 25.
freq_bins = 4  # how many frequency bins to use

epochs = Epochs(raw, events, event_id, tmin, tmax,
                    proj=False, baseline=None, add_eeg_ref=False, preload=True)
y = epochs.events[:, 2] - 2
sfreq = epochs.info['sfreq']

tf = TimeFrequencyCSP(tmin, tmax, min_freq, max_freq, freq_bins, cv, estimator,
                      sfreq=sfreq)
tf.fit(epochs, y)
scores = tf.score(epochs, y)
