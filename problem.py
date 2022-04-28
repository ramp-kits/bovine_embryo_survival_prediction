# Dependencies

import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
import rampwf as rw
from rampwf.utils.importing import import_module_from_source


# --------------------------------------------------
#
# Challenge title

problem_title = "Bovine embryos survival prediction"


# --------------------------------------------------
#
# Select Prediction type

_prediction_label_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

pred_times = [27, 32, 37, 40, 44, 48, 53, 58, 63, 94, None]
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names * len(pred_times)
)

# --------------------------------------------------
#
# Select Workflow


def _cut_video(video, pred_time: float):
    """
    Utility function to modify a VideoReader object by cutting its frame_times
    attribute in order to access only times <= pred_time
    :param video A VideoReader object
    :param pred_time: float the desired time of the cut
    :return: The VideoReader object with modified frame_times attribute
    """
    video = copy.copy(video)
    if pred_time is None:
        pred_time = video.frame_times[-1]
    video.frame_times = video.frame_times[video.frame_times <= pred_time]
    video.nb_frames = len(video.frame_times)

    return video


class SequenceClassifierWorkflow:
    """
    RAMP workflow for the BovMovies2Pred challenge. Its intended use is for
    training different models at different prediction times.

    Submissions need to contain one file: estimator.py, with the following
    requirements:
        - videoclassifier.py  - submitted function
            - class VideoClassiffier  - estimator to train
                - def fit(videos, y_true, pred_time)  - defined method for
                training on videos cut at time `pred_time`
                - def predict(videos, pred_time)      - defined method for
                predicting on videos cut at time `pred_time`
    """

    def __init__(self, workflow_element_names: list = ["videoclassifier.py"]):
        """
        Parameters
        ----------
        workflow_element_names : list [str]
            List of the names for the elements of the workflow. Included to be
            consistent with RAMP API.
        """
        self.element_names = workflow_element_names
        self.estimators = None
        return

    def train_submission(
        self,
        module_path: str,
        X_array: list,
        y_array: list,
        train_is: list = None,
    ):
        """Trains the submitted estimator.

        Parameters
        ----------
        module_path : str
            Leading path to the user's custom modules (typically submissions/X)
        X_array : list
            List of VideoReader.
        y_array : list
            List of labels.
        train_is : list
            List of indices indicating the entries in X_array to use for
            training.

        Returns
        -------
        estimators
            List of trained estimator on each time.
        """

        if train_is is None:
            train_is = slice(None, None, None)

        estimator_module = import_module_from_source(
            os.path.join(module_path, self.element_names[0]),
            self.element_names[0],
            sanitize=True,
        )
        self.estimators = []

        for pred_time in pred_times:
            model = estimator_module.VideoClassifier()
            videos = [_cut_video(X_array[idx], pred_time) for idx in train_is]
            # Fit a model at time `pred_time`
            model.fit(videos, y_array[train_is], pred_time=pred_time)
            self.estimators.append(model)

        return self.estimators

    def test_submission(self, trained_estimators: list, X_array: list):
        """Test submission

        Parameters
        ----------
        trained_estimator: list
            List of models previously trained by train_submission
        X_array : list
            List of length n_samples, containing VideoReader objects on which
            to make the predictions.

        Returns
        -------
        pred: numpy.array
            An array of size (n_times * n_samples, n_classes)

        """
        preds = []
        for t, pred_time in enumerate(pred_times):
            videos = [_cut_video(video, pred_time) for video in X_array]
            preds.append(trained_estimators[t].predict(videos, pred_time))
        preds = np.concatenate(preds, axis=1)
        return preds


workflow = SequenceClassifierWorkflow()

# --------------------------------------------------
#
# Define the score types
# Custom loss implementation
# See README of:
# https://github.com/paris-saclay-cds/ramp-workflow/tree/\
# master/rampwf/score_types


class WeightedClassificationError(rw.score_types.BaseScoreType):
    """
    Classfification error with expert-designed weight.

    Some errors (e.g. predicting class "H" when it is class "A") might count
    for more in the final scores. The missclassification weights were
    designed by an expert.
    """

    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf  # 1 if normalisation by max(W)

    def __init__(
        self, name="WeightedClassificationError", precision=2, time_idx=0
    ):
        self.name = name + f"[{time_idx + 1}]"
        self.precision = precision
        self.time_idx = time_idx

    def compute(self, y_true, y_pred):

        n_classes = len(_prediction_label_names)

        # missclassif weights matrix
        W = np.array(
            [
                [0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],
            ]
        )
        W = W / np.max(W)

        # Convert proba to hard-labels
        y_pred = np.argmax(y_pred, axis=1)

        n = len(y_true)
        conf_mat = confusion_matrix(
            y_true, y_pred, labels=np.arange(n_classes)
        )
        loss = np.multiply(conf_mat, W).sum() / n
        return loss

    def __call__(self, y_true, y_pred):
        n_classes = len(_prediction_label_names)

        # select the prediction corresponding to time_idx
        y_pred = y_pred[
            :,
            self.time_idx
            * n_classes : (self.time_idx + 1)  # noqa:E203
            * n_classes,  # noqa:E203
        ]

        # Convert proba to hard-labels
        y_true = y_true[:, :n_classes]
        y_true = np.argmax(y_true, axis=1)

        return self.compute(y_true, y_pred)


score_types = [
    WeightedClassificationError(name="WeightedClassifErr", time_idx=time_idx)
    for time_idx in range(len(pred_times))
]

# --------------------------------------------------
# CV scheme


def get_cv(X, y):
    cv = ShuffleSplit(
        n_splits=2,
        train_size=0.8,
        random_state=42,
    )
    return cv.split(X, y)


# --------------------------------------------------
# I/O methods
#
# custom video reader based on opencv


class VideoReader:
    def __init__(self, video_filename, frame_times, img_size=[250, 250]):
        import cv2

        self.video = cv2.VideoCapture(video_filename)
        self.nb_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.img_size = img_size
        self.frame_times = frame_times

    def read_frame(self, frame_time):

        """Return the frame of a VideoReader object at the specified
        `frame_time`

        Args:
            frame_time (float): the specified time in hours (allowing quarter
            hours, e.g. 25.75 or 26.50)

        Raises:
            ValueError: If the specified time does not exist for the selected
            video

        Returns:
            np.ndarray: A 2-D array containing the grayscale image.
        """
        import cv2

        if frame_time is None:
            frame_time = self.frame_times[-1]
        elif frame_time not in self.frame_times:
            raise ValueError(
                "The specified frame time must me within the time "
                "interval of the video."
            )

        frame_nb = np.where(self.frame_times == frame_time)[0][0]
        if frame_nb is not None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
        _, frame = self.video.read()

        # always reset video's frame counter to 0 to avoid unexpected behavior
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def read_sequence(self, begin_time=None, end_time=None):

        """Extract the sequence of consecutive frames from begin_time to
        end_time (included).

        Args:
            begin_time (float, optional): The time where the extraction begins.
            Defaults to None.
            end_time (float, optional):  The time where the extraction ends.
            *Defaults to None.

        Returns:
            np.ndarray: A 3-D numpy array with first axis corresponding to the
            frame index and the remaining dimension to image size.
        """
        import cv2

        if begin_time is None:
            begin_time = self.frame_times[0]
        elif begin_time not in self.frame_times:
            raise ValueError(
                "The specified begin_time must me within the time"
                " interval of the video."
            )

        if end_time is None:
            end_time = self.frame_times[-1]
        elif end_time not in self.frame_times:
            raise ValueError(
                "The specified pred_time must me within the time "
                "interval of the video."
            )

        if begin_time > end_time:
            raise ValueError("begin_time must be smaller than pred_time.")

        begin_nb = np.where(self.frame_times == begin_time)[0][0]
        end_nb = np.where(self.frame_times == end_time)[0][0]
        self.video.set(cv2.CAP_PROP_POS_FRAMES, begin_nb)

        my_frames = list(range(begin_nb, end_nb + 1))
        video_array = np.empty(
            shape=(len(my_frames), self.img_size[0], self.img_size[1])
        )
        for t, _ in enumerate(my_frames):
            _, frame = self.video.read()
            video_array[t, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # always reset video's frame counter to 0 to avoid unexpected behavior
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return video_array

    def read_samples(self, selected_times=None):

        """Read several frames of the video at once corresponding to the
        selected times.

        Args:
            selected_times (list, optional): The list of of desired extraction
            times, in hours (allowing quarter hour). Defaults to None, the
            whole 300 frames are returned.

        Returns:
            np.ndarray: A 3-D numpy array with of shape
            (size len(selected_times), 250, 250).
        """
        import cv2

        if selected_times is None:
            selected_times = self.frame_times

        res = np.empty(
            [len(selected_times), self.img_size[0], self.img_size[1]]
        )
        frame_nbs = np.where([t in selected_times for t in self.frame_times])[
            0
        ]
        for i, f in enumerate(frame_nbs):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, f)
            _, frame = self.video.read()

            res[i, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return res


def _read_data(path, dir_name, classification="class"):
    metadata = pd.read_csv(
        os.path.join(path, "data", dir_name, dir_name + "_metadata.csv")
    )
    metadata["video_filename"] = metadata.name + ".mp4"
    labels = metadata[classification].values

    # recalage des temps des vidéos
    frame_times = np.zeros(shape=(len(labels), 300))
    for k, t0 in enumerate(metadata.t0.values):
        frame_times[k, :] = t0 + np.array([x * 0.25 for x in range(300)])

    videos = []
    for k, file in enumerate(metadata.video_filename):
        videos.append(
            VideoReader(
                os.path.join(path, "data", dir_name, file),
                frame_times=frame_times[k, :],
            )
        )

    if os.getenv("RAMP_TEST_MODE", 0):
        videos, labels = videos[:30], labels[:30]

    return videos, labels


def get_train_data(path="."):
    dir_name = "train"
    return _read_data(path, dir_name)


def get_test_data(path="."):
    f_name = "test"
    return _read_data(path, f_name)


if __name__ == "__main__":
    import rampwf

    os.environ["RAMP_TEST_MODE"] = "1"
    rampwf.utils.testing.assert_submission()
