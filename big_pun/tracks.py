import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Tracks:
    def __init__(self, track_inits):
        self.all_ids = np.array(list(track_inits.keys()), dtype=int)
        self.all_features = np.stack([track_inits[i] for i in self.all_ids])
        self.t = np.min(self.all_features[:,0,0])

        close = np.abs(self.all_features[:,0,0] - self.t) < 1e-4
        self.active_ids = self.all_ids[close]
        self.active_features = np.stack([track_inits[i][:,1:] for i in self.active_ids]).astype(np.float32)

        self.tracks_dict = track_inits

    def update(self, mask, tracks, t):
        # update tracks
        self.t = t
        self.active_features = tracks

        # remove all deletes
        self.active_features = self.active_features[mask]
        self.active_ids = self.active_ids[mask]

        # append to dict
        for i, feature in zip(self.active_ids, self.active_features):
            self.tracks_dict[i] = np.concatenate([self.tracks_dict[i],
                                                  np.concatenate([[[t]], feature], 1)], 0)

        # add new tracks
        # get close features inits
        close = np.abs(self.all_features[:, 0, 0] - t) < 1e-4
        new_ids = self.all_ids[close]
        new_tracks = self.all_features[close, :, 1:].astype(np.float32)

        self.active_ids = np.concatenate([self.active_ids, new_ids])
        self.active_features = np.concatenate([self.active_features, new_tracks])

    def collate(self):
        collated_features = None
        for i, vals in self.tracks_dict.items():
            id_block = np.concatenate([np.ones((len(vals), 1)) * i, vals], 1)

            if collated_features is None:
                collated_features = id_block
            else:
                collated_features = np.concatenate(
                    [collated_features, id_block])

        # sort by timestamps
        collated_features = collated_features[collated_features[:, 1].argsort(
        )]
        return collated_features

    def get_ref_tracks_before_t(self, tracks, t, ids):
        ref_tracks = []
        lost_ids = []
        tracks = tracks[(tracks[:, 1] <= t + 1e-3) & (tracks[:, 1] > t - 0.1)]
        for i in ids:
            track_before_t = tracks[tracks[:,0] == i]
            if len(track_before_t) > 0:
                ref_tracks.append(track_before_t[-1, 2:])
            else:
                lost_ids.append(i)
        if len(ref_tracks) > 0:
            return np.stack(ref_tracks).reshape((-1,1,2)), lost_ids
        return None, lost_ids

    def viz(self, img, handles=None, ref_tracks=None, wait_ms=1000.0):
        lost_ids = []
        if ref_tracks is not None:
            ref_tracks, lost_ids = self.get_ref_tracks_before_t(ref_tracks, self.t, self.active_ids)

        # filter out lost ids from gt
        mask = np.full(self.active_ids.shape, True, dtype=bool)
        for i in lost_ids:
            mask = mask & (self.active_ids != i)
        active_tracks = self.active_features[mask]
        # take out all tracks that are in plot_idx
        if handles is None:
            fig, ax = plt.subplots()

            handles = []
            handles += [ax.imshow(img)]
            handles += [ax.plot(active_tracks[:, 0, 0], active_tracks[:, 0, 1],
                                color='b', marker='x', ls="", ms=4, label="gt")]
            handles += [fig]

            if ref_tracks is not None:
                handles += [ax.plot(ref_tracks[:, 0, 0], ref_tracks[:, 0, 1],
                                    color='g', marker='x', ls="", ms=4, label="reference")]

            ax.legend()

            plt.title("T = %.4f" % self.t)
            plt.show(block=False)
        else:
            plt.title("T = %.4f" % self.t)
            handles[0].set_data(img)
            handles[1][0].set_data(active_tracks[:, 0, 0], active_tracks[:, 0, 1])
            handles[2].canvas.draw()

            if ref_tracks is not None:
                handles[3][0].set_data(ref_tracks[:, 0, 0], ref_tracks[:, 0, 1])

        plt.pause(wait_ms/1000.0)

        return handles
