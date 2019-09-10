import yaml
from dataset import Dataset
import numpy as np
from tracker_utils import backProjectFeatures, grid_sample, filter_first_tracks
from tracks import Tracks
import os


class TrackerInitializer:
    def __init__(self, root, dataset_yaml, config_yaml=None, config=None):
        if config is None:
            with open(config_yaml, "r") as f:
                self.config = yaml.load(f, Loader=yaml.Loader)
        else:
            self.config = config

        print("Constructing initializer with type '%s'" % self.config["type"])

        if self.config["type"] == "tracks":
            self.tracks_obj, self.tracker_params = self.init_on_track(root, self.config, dataset_yaml)
        elif self.config["type"] == "depth_from_tracks":
            self.tracks_obj, self.tracker_params = self.init_by_reprojection(root, self.config, dataset_yaml)
        else:
            raise ValueError

        print ("Done")

    def init_by_reprojection(self, root, config, dataset_yaml):
        """
        inits tracks by extracting corners on image and then backprojecting them for a given pose.
        """
        print("[1/3] Loading tracks in %s" % os.path.basename(config["tracks_csv"]))
        tracks = np.genfromtxt(config["tracks_csv"])
        first_len_tracks = len(tracks)
        valid_ids, tracks = filter_first_tracks(tracks, filter_too_short=True)

        if len(tracks) < first_len_tracks:
            print("WARNING: This package only supports evaluation of tracks which have been initialized at the same"
                  "time. All tracks except the first have been discarded.")

        tracks_dict = {i: tracks[tracks[:, 0] == i, 1:] for i in valid_ids}
        features = np.stack([tracks_dict[i][0] for i in valid_ids]).reshape(-1, 1, 3)

        print("[2/3] Loading depths, poses, frames and camera info")
        depth_dataset = Dataset(root, dataset_yaml, dataset_type="depth")
        pose_dataset = Dataset(root, dataset_yaml, dataset_type="poses")
        camera_info_dataset = Dataset(root, dataset_yaml, dataset_type="camera_info")

        K = camera_info_dataset.K
        img_size = camera_info_dataset.img_size

        # find poses and depths at features
        print("[3/3] Backprojecting first feature positions")
        depth_maps_interp = depth_dataset.get_interpolated(features[:, 0, 0])
        depths = grid_sample(features[:,0,1:], depth_maps_interp)

        poses = pose_dataset.get_interpolated(features[:, 0, 0])
        landmarks = backProjectFeatures(K, features[:, :, 1:], depths, poses)

        t_min = np.min(features[:, 0, 0])
        pose_dataset.set_to_first_after(t_min)

        # parse
        landmarks_dict = {i: landmarks[j] for j,i in enumerate(valid_ids)}
        features_dict = {i: features[j] for j,i in enumerate(valid_ids)}

        # create dict of features
        tracks_obj = Tracks(features_dict)

        # params for tracker
        tracker_params = {"landmarks": landmarks_dict,
                          "pose_dataset": pose_dataset, "img_size": img_size, "K": K, "reference_track": tracks}

        return tracks_obj, tracker_params

    def init_on_track(self, root, config, dataset_yaml):
        print("[1/3] Loading tracks in %s." % os.path.basename(config["tracks_csv"]))
        tracks = np.genfromtxt(self.config["tracks_csv"])

        # check that all features start at the same timestamp, if not, discard features that occur later
        first_len_tracks = len(tracks)
        valid_ids, tracks = filter_first_tracks(tracks, filter_too_short=True)

        if len(tracks) < first_len_tracks:
            print("WARNING: This package only supports evaluation of tracks which have been initialized at the same"
                  "time. All tracks except the first have been discarded.")

        tracks_dict = {i: tracks[tracks[:, 0] == i, 1:] for i in valid_ids}

        print("[2/3] Loading frame dataset to find positions of initial tracks.")
        frame_dataset = Dataset(root, dataset_yaml, dataset_type="frames")

        # find dataset indices for each start
        tracks_init = {}
        print("[3/3] Initializing tracks")
        for track_id, track in tracks_dict.items():
            frame_dataset.set_to_first_after(track[0,0])
            t_dataset, _ = frame_dataset.current()

            x_interp = np.interp(t_dataset, track[:, 0], track[:, 1])
            y_interp = np.interp(t_dataset, track[:, 0], track[:, 2])

            tracks_init[track_id] = np.array([[t_dataset, x_interp, y_interp]])

        tracks_obj = Tracks(tracks_init)

        return tracks_obj, {"frame_dataset": frame_dataset, "reference_track": tracks}

