"""
File that takes dataset.yaml path and generates an object containing images and timestamps
"""
import yaml
import numpy as np
from os.path import join, isfile

from tracker_utils import slerp, q_to_R, interpolate
from bag2dataframe import Bag2Images, Bag2DepthMap, Bag2Trajectory, Bag2CameraInfo


class Dataset:
    def __init__(self, root, dataset_yaml, dataset_type="images"):
        self.root = root
        self.dataset_yaml = dataset_yaml
        self.dataset_type = dataset_type

        if dataset_type == "frames":
            load = self.load_images
        elif dataset_type == "poses":
            load = self.load_poses
        elif dataset_type == "depth":
            load = self.load_depth
        elif dataset_type == "camera_info":
            load = self.load_camera_info
        else:
            raise ValueError

        self.queue, self.times, self.config = load(root, dataset_yaml)

        self.it = 0

    def _load(self, root, dataset_yaml, topic, cls, *attrs):
        """
        load
        """
        with open(dataset_yaml, "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        path = join(root, config["name"])
        assert isfile(path), "The dataset '%s' does not exist." % join(root, config["name"])

        if config["type"] == "bag":
            assert topic in config, "The dataset config at '%s' needs a key with name '%s'." % (dataset_yaml, topic)
            bag = cls(path, topic=config[topic])

            for attr in attrs:
                setattr(self, attr, getattr(bag, attr))

            if not hasattr(bag, "times"):
                return getattr(bag, attrs[0]), None, config
            return getattr(bag, attrs[0]), np.array(bag.times), config

    def load_camera_info(self, root, dataset_yaml):
        return self._load(root, dataset_yaml, "camera_info_topic", Bag2CameraInfo, "K", "img_size")

    def load_poses(self, root, dataset_yaml):
        return self._load(root, dataset_yaml, "pose_topic", Bag2Trajectory, "poses", "quaternions")

    def load_images(self, root, dataset_yaml):
        return self._load(root, dataset_yaml, "image_topic", Bag2Images, "images")

    def load_depth(self, root, dataset_yaml):
        return self._load(root, dataset_yaml, "depth_map_topic", Bag2DepthMap, "depth_maps")

    def interpolate_pose(self, t):
        """
        interpolate pose using slerp
        """
        concatenated_poses = np.stack(self.queue)
        T = concatenated_poses[:,:3,3]

        # convert R to q
        q = np.stack(self.quaternions)

        q_interp = slerp(t, self.times, q)

        # find R
        R_interp = q_to_R(q_interp)

        # find T
        T_interp = interpolate(t, self.times, T)

        # put together into N x 4 x 4 matrix
        interp_pose = np.zeros((T_interp.shape[0], 4, 4))
        interp_pose[:,:3,3] = T_interp
        interp_pose[:,:3,:3] = R_interp
        interp_pose[:,3,3] = 1

        return interp_pose 

    def get_interpolated(self, t):
        """
        get interpolated value
        """
        if self.dataset_type == "poses":
            return self.interpolate_pose(t)

        values = np.stack(self.queue)
        values_interp = interpolate(t, self.times, values)

        return values_interp

    def current(self):
        return self[self.it]

    def set_to_first_after(self, t):
        assert t <= self.times[-1]
        mask = (self.times - t >= 0) | (np.abs(self.times - t) < 1e-5)
        self.it = np.where(mask)[0][0]

    def get_first_after(self, t):
        assert t <= self.times[-1]
        mask = (self.times - t >= 0) | (np.abs(self.times - t) < 1e-5)
        return self[np.where(mask)[0][0]]

    def get_closest(self, t):
        idx = np.argmin(np.abs(self.times - t)).astype(int)
        return self[idx]

    def __getitem__(self, idx):
        return self.times[idx], self.queue[idx]

    def __len__(self):
        return len(self.queue) - self.it

    def __iter__(self):
        return self

    def next(self):
        if self.it == len(self.queue):
            raise StopIteration
        else:
            self.it += 1
            return self[self.it-1]