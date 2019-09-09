import cv2
import numpy as np
from tracker_utils import projectLandmarks
import tqdm


class Tracker:
    def __init__(self, config):
        self.config = config

        if self.config["type"] == "KLT":
            assert "window_size" in self.config, "The tracker config of type KLT needs the key window_size."
            assert "num_pyramidal_layers" in self.config, "The tracker config of type KLT needs the key num_pyramidal_layers."
            self.track = self.track_features_on_klt
        elif self.config["type"] == "reprojection":
            self.track = self.track_features_with_landmarks
        else:
            raise ValueError

    def track_features_on_klt(self, tracks_obj, tracker_params):
        """
        tracks features in feature_init using the dataset
        tracks must be dict with keys as ids and values as 1 x 3 array with x,y,t
        returns a dict with keys as track ids, and values as N x 3 arrays, with x,y,t.
        If collate is true, returns N x 4 array with id,x,y,t .
        """
        assert "reference_track" in tracker_params
        assert "frame_dataset" in tracker_params

        window_size = self.config["window_size"]
        num_pyramidal_layers = self.config["num_pyramidal_layers"]

        dataset = tracker_params["frame_dataset"]
        dataset.set_to_first_after(tracks_obj.t)

        print("Tracking with KLT parameters: [window_size=%s num_pyramidal_layers=%s]" % (window_size, num_pyramidal_layers))
        for i, (t, img) in enumerate(tqdm.tqdm(dataset)):
            if i == 0:
                first_img = img
                continue

            second_img = img
            
            if len(tracks_obj.active_features) == 0:
                break
                
            new_features, status, err = \
                cv2.calcOpticalFlowPyrLK(first_img, second_img, tracks_obj.active_features,
                                         None, winSize=(window_size, window_size), maxLevel=num_pyramidal_layers)

            tracks_obj.update(status[:,0]==1, new_features, t)

            first_img = second_img

        tracks = tracks_obj.collate()
        return tracks

    def track_features_with_landmarks(self, tracks_obj, tracker_params):
        """
        Track features feature_init by projecting it onto frames at every time step.
        dataset here is a pose dataset
        """
        # get camera calibration
        assert "img_size" in tracker_params
        assert "K" in tracker_params
        assert "landmarks" in tracker_params
        assert "pose_dataset" in tracker_params

        W, H= tracker_params["img_size"]
        K = tracker_params["K"]
        landmarks = tracker_params["landmarks"]
        pose_dataset = tracker_params["pose_dataset"]

        print("Tracking via reprojection")
        for i, (t, pose) in enumerate(tqdm.tqdm(pose_dataset)):
            # get closest image
            if i == 0:
                continue

            # all features lost
            if len(tracks_obj.active_ids) == 0:
                break

            active_landmarks = np.stack([landmarks[i] for i in tracks_obj.active_ids])
            new_features = projectLandmarks(active_landmarks, pose, K)

            x_new, y_new = new_features[:, 0, 0], new_features[:, 0, 1]
            status = (x_new >= 0) & (y_new >= 0) & (x_new < W) & (y_new < H)

            tracks_obj.update(status, new_features, t)

        tracks = tracks_obj.collate()

        return tracks

