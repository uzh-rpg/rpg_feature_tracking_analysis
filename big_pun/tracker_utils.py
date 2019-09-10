import numpy as np
import tqdm


def backProjectFeatures(K, features, depths, poses):
    assert len(features.shape) == 3
    assert features.shape[1] == 1
    assert features.shape[2] == 2
    assert len(K.shape) == 2
    assert K.shape[0] == 3
    assert K.shape[1] == 3
    assert len(depths.shape) == 1
    assert depths.shape[0] == features.shape[0]
    assert len(poses.shape) == 3
    assert poses.shape[1] == 4
    assert poses.shape[2] == 4

    features_x_camera = depths * (features[:,0,0] - K[0, 2]) / K[0, 0]
    features_y_camera = depths * (features[:,0,1] - K[1, 2]) / K[1, 1]

    landmarks_local_hom = np.stack([features_x_camera, features_y_camera, depths, np.ones(len(depths))]).T.reshape(-1, 4, 1)
    landmarks_global_hom = np.matmul(poses, landmarks_local_hom)

    landmarks_global = landmarks_global_hom[:, :3,0] / landmarks_global_hom[:, 3:4,0]

    return landmarks_global

def projectLandmarks(landmarks_global, pose, K):
    assert len(landmarks_global.shape) == 2
    assert landmarks_global.shape[1] == 3
    assert len(pose.shape) == 2
    assert pose.shape[0] == 4
    assert pose.shape[1] == 4
    assert len(K.shape) == 2
    assert K.shape[0] == 3
    assert K.shape[1] == 3

    inv_pose = np.linalg.inv(pose)
    ones = np.ones((landmarks_global.shape[0], 1))
    landmarks_global_hom = np.concatenate([landmarks_global, ones], 1)
    landmarks_global_hom = landmarks_global_hom.reshape(-1, 4, 1)

    landmarks_local_hom = np.matmul(inv_pose, landmarks_global_hom)
    landmarks_local_hom = landmarks_local_hom[:, :3, :] / landmarks_local_hom[:, 3:4, :]

    features_hom = np.matmul(K, landmarks_local_hom)
    features = features_hom[:, :2, 0] / features_hom[:, 2:3, 0]

    features = features.reshape((-1, 1, 2))
    return features

def getTrackData(path, delimiter=" ", filter_too_short=False):
    data = np.genfromtxt(path, delimiter=delimiter)
    valid_ids, data = filter_first_tracks(data, filter_too_short)
    track_data = {i: data[data[:,0]==i, 1:] for i in valid_ids}
    return track_data

def filter_first_tracks(tracks, filter_too_short=False):
    tmin = tracks[0, 1]
    valid_ids = np.unique(tracks[tracks[:, 1] == tmin, 0]).astype(int)
    all_ids = np.unique(tracks[:, 0]).astype(int)
    for id in all_ids:
        if id not in valid_ids:
            tracks = tracks[tracks[:, 0] != id]
        else:
            if filter_too_short:
                num_samples = len(tracks[tracks[:,0]==id])
                if num_samples < 3:
                    tracks = tracks[tracks[:, 0] != id]
                    valid_ids = valid_ids[valid_ids!=id]

    return valid_ids, tracks

def getError(est_data, gt_data):
    # discard gt which happen after last est_data
    gt_data = gt_data[gt_data[:, 0] <= est_data[-1, 0]]

    est_t, est_x, est_y = est_data.T
    gt_t, gt_x, gt_y = gt_data.T

    if np.abs(gt_t[0] - est_t[0]) < 1e-5:
        gt_t[0] = est_t[0]

    if len(est_t) < 2:
        return gt_t, np.array([0]), np.array([0])

    # find samples which have dt > threshold
    error_x = np.interp(gt_t, est_t, est_x) - gt_x
    error_y = np.interp(gt_t, est_t, est_y) - gt_y

    return gt_t, error_x, error_y

def compareTracks(est_track_data, gt_track_data):
    error_data = np.zeros(shape=(0, 4))

    for track_id, est_track in tqdm.tqdm(est_track_data.items()):
        gt_track = gt_track_data[track_id]

        # interpolate own track at time points given in gt track
        gt_t, e_x, e_y = getError(est_track, gt_track)

        if len(gt_t) != 0:
            ids = (track_id * np.ones_like(e_x)).astype(int)
            added_data = np.stack([ids, gt_t, e_x, e_y]).T
            error_data = np.concatenate([error_data, added_data])

    # sort times
    error_data = error_data[error_data[:, 1].argsort()]

    return error_data

def q_to_R(q):
    if len(q.shape) == 1:
        q = q[None, :]

    w, x, y, z = q.T
    return np.dstack([w ** 2 + x ** 2 - y ** 2 - z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, \
                     2 * x * y + 2 * w * z, w ** 2 - x ** 2 + y ** 2 - z ** 2, 2 * y * z - 2 * w * x, \
                     2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w ** 2 - x ** 2 - y ** 2 + z ** 2]).reshape(len(w), 3, 3)

def get_left_right_dt(tq, ts, vs):
    left_index = np.clip(np.searchsorted(ts, tq) - 1, 0, len(ts) - 1)
    right_index = np.clip(left_index + 1, 0, len(ts) - 1)
    dt = (tq - ts[left_index]) / (ts[right_index] - ts[left_index])

    left_q = vs[left_index]
    right_q = vs[right_index]

    return left_q, right_q, dt

def slerp(tq, ts, qs):
    left_q, right_q, dt = get_left_right_dt(tq, ts, qs)

    # perform slerp
    omega = np.arccos((left_q * right_q).sum(1))
    omega[omega==0] = 1e-4
    so = np.sin(omega)
    q_interp = (np.sin((1.0 - dt) * omega) / so)[:,None] * left_q + (np.sin(dt * omega) / so)[:,None] * right_q
    q_interp /= (q_interp ** 2).sum(1, keepdims=True)

    return q_interp

def interpolate(tq, ts, Ts):
    left_Ts, right_Ts, dt = get_left_right_dt(tq, ts, Ts)

    rank = len(Ts.shape)
    dt = dt.reshape(-1, *((rank-1)*[1]))

    T_interp = (1-dt) * left_Ts + dt * right_Ts

    return T_interp

def grid_sample(positions, depth_maps):
    D, H, W = depth_maps.shape

    x_left, y_down = positions.astype(int).T
    y_up = np.clip(y_down + 1, 0, H - 1)
    x_right = np.clip(x_left + 1, 0, W - 1)

    r_x, r_y = positions.T - positions.T.astype(int)

    d_index = np.arange(D).astype(int)

    d_up_left = depth_maps[d_index, y_up, x_left]
    d_down_left = depth_maps[d_index, y_down, x_left]
    d_up_right = depth_maps[d_index, y_up, x_right]
    d_down_right = depth_maps[d_index, y_down, x_right]

    d_interp = r_x*r_y*d_down_right + (1-r_x)*r_y*d_down_left + r_x*(1-r_y)*d_up_right + (1-r_x)*(1-r_y)*d_up_left

    return d_interp

def mean_filtering(data, ts, kernel=None, side="both"):
    assert len(data) == len(ts)
    if kernel is None:
        kernel = 1.5 * np.max(np.diff(ts))

    smooth_ts = []
    smooth_data = []

    t = 0

    while True:
        if side == "left":
            samples = data[(ts<=t) & (ts>t-kernel)]
        elif side == "right":
            samples = data[(ts>=t) & (ts<t+kernel)]
        elif side == "both":
            samples = data[(ts>=t-kernel) & (ts<t+kernel)]

        if len(samples) == 0:
            break

        smooth_data.append(np.mean(samples))
        smooth_ts.append(t)

        t += (2*kernel)

    return smooth_data, smooth_ts


