import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import yaml
from os.path import isfile, join, isdir
from big_pun.tracker_utils import mean_filtering


def check_config(config, root, config_path):
    assert type(config) is list, "Individual method labels in '%s' need to be listed." % config
    for element in config:
        assert type(element) is dict, "Individual methods in '%s' should be dictionaries." % config_path
        value = element.values()[0]
        assert len(value) == 2, "Individual methods in '%s' should contain only color and folder location." % config_path
        dest_folder = join(root, value[1])
        assert isdir(dest_folder), "Folder %s specified in config must exist." % dest_folder
        assert type(value[0]) is list and len(value[0]) == 3, "Color '%s' must be a list of RGB values." % value[0]
        tracks_file = join(dest_folder, "tracks.txt")
        assert isfile(tracks_file), "Tracks '%s' do not exist." % tracks_file
        gt_file = join(dest_folder, "tracks.txt.gt.txt")
        assert isfile(gt_file), "Ground truth '%s' does not exist." % gt_file
        error_file = join(dest_folder, "tracks.txt.errors.txt")
        assert isfile(error_file), "Error file '%s' does not exist." % error_file

def loadMethod(directory):
    # load and parse errors
    f_errors = os.path.join(directory, "tracks.txt.errors.txt")
    errors = np.genfromtxt(f_errors)
    error_ids = np.unique(errors[:, 0])

    # load and parse tracks
    f_tracks = os.path.join(directory, "tracks.txt")
    tracks = np.genfromtxt(f_tracks)
    track_ids = np.unique(tracks[:, 0])

    # check that errors and error ids are correct
    assert len(set(track_ids).symmetric_difference(error_ids)) == 0, \
        "Ids of tracks and error file are not compatible for '%s' and '%s'." % (f_tracks, f_errors)

    common_ids = set(track_ids).intersection(set(error_ids))

    tracks_dict = {i: tracks[tracks[:, 0] == i, 1:] for i in common_ids}
    errors_dict = {i: errors[errors[:, 0] == i, 1:] for i in common_ids}

    return errors_dict, tracks_dict

def plot_tracks_3d(frame_dataset, est_file, gt_file=None, t_max=-1, method="estimation"):
    """
    Plots 3d. options are
    1. plot gt or not
    2. set t_max
    3. plot only first features

    :param frame_dataset:
    :param est_file:
    :param gt_file:
    :param t_max:
    :param first_features:
    :return:
    """
    est_tracks_data = np.genfromtxt(est_file)
    plot_ids = np.unique(est_tracks_data[:, 0])

    plot_features_dict = {i: est_tracks_data[est_tracks_data[:,0]==i, 1:] for i in plot_ids}

    # if gt file is known also load it.
    if gt_file is not None:
        gt_tracks_data = np.genfromtxt(gt_file)
        plot_gt_dict = {i: gt_tracks_data[gt_tracks_data[:,0]==i, 1:] for i in plot_ids}

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot image
    t_min = np.min(est_tracks_data[:, 1])
    t_img, img = frame_dataset.get_closest(t_min)
    img = img.astype(float) / 255

    H, W, _ = img.shape
    X = np.arange(0, W)
    Y = np.arange(0, H)
    X, Y = np.meshgrid(X, Y)

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 0.8, 0.8 * float(H) / float(W), 1]))

    ax.plot_surface(t_img - t_min, X, Y, cstride=1, rstride=1, facecolors=img, linewidth=1, shade=False, edgecolor='none')

    # formatting
    ax.view_init(elev=20)
    ax.invert_zaxis()

    if t_max != -1:
        ax.set_xlim([0, t_max])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax.w_xaxis.line.set_linewidth(1.5)
    ax.w_yaxis.line.set_linewidth(1.5)
    ax.w_zaxis.line.set_linewidth(1.5)
    ax.get_xaxis().set_visible(False)
    ax.xaxis.labelpad = 20
    ax.get_yaxis().set_visible(False)
    ax.zaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlabel("Time [s]", fontsize=15)
    ax.set_ylim([0, W - 1])
    ax.set_zlim([H - 1, 0])

    # plot tracks
    for i, (track_id, track) in enumerate(plot_features_dict.items()):
        # limit to tmax
        track[:,0] -= t_min
        if t_max != -1:
            track = track[track[:,0]<=t_max]
        t, x, y = track.T

        if gt_file is not None:
            if i == 0:
                ax.plot(t, x, y, linewidth=1.5, color="b", label=method)
            else:
                ax.plot(t, x, y, linewidth=1.5, color="b")

            gt_track = plot_gt_dict[track_id]
            gt_track[:,0] -= t_min
            if t_max != -1:
                gt_track = gt_track[gt_track[:,0]<=t_max]

            gt_track = gt_track[gt_track[:,0]<=t[-1]]
            t_gt, x_gt, y_gt = gt_track.T

            if i == 0:
                ax.plot(t_gt, x_gt, y_gt, linewidth=1.5, color="g", label="ground truth")
            else:
                ax.plot(t_gt, x_gt, y_gt, linewidth=1.5, color="g")

        else:
            ax.plot(t, x, y, linewidth=1.5)

    if gt_file is not None:
        ax.legend(frameon=False)

    return fig, ax

def plot_num_features(f, config_path=None, root=None, error_threshold=10, method="estimation", color=[0,150,0]):
    plt.rc('text', usetex=True)
    plt.rc('axes', linewidth=3)

    root = root or os.path.dirname(f)

    print("Parsing config.")
    if config_path is not None:
        with open(config_path, "r") as f:
            try:
                config = yaml.load(f)
            except:
                raise Exception("Yaml file '%s' is not formatted correctly." % config_path)
    else:
        config = [{method: [color, root]}]

    check_config(config, root, config_path)

    fig, ax = plt.subplots()

    # parse config
    labels = [v.keys()[0] for v in config]
    files = [v.values()[0][1:] for v in config]
    colors = [v.values()[0][0] for v in config]
    colors = [[float(c) / 255 for c in row] for row in colors]

    print("Found the following methods: %s" % labels)

    # load our method for reference
    if error_threshold > 0:
        print("Filtering tracks with error threshold = %s" % error_threshold)

    summary = {}

    print("Preparing plot")
    # formatting
    for label, folders, color in zip(labels, files, colors):
        print("Processing method %s in folder %s." % (label, folders[0]))
        assert len(folders) > 0

        # load and parse errors
        errors_dict, tracks_dict = loadMethod(os.path.join(root, folders[0]))

        # filter tracks and errors when they exceed a threshold
        for i, track in tracks_dict.items():
            error = errors_dict[i]
            if error_threshold > 0:
                error_euclidean = (error[:, 1] ** 2 + error[:, 2] ** 2) ** .5
                idxs = np.where(error_euclidean > error_threshold)[0]

                if len(idxs) != 0:
                    error = error[:idxs[0]]
                    if len(error) == 0:
                        errors_dict[i] = np.array([])
                        tracks_dict[i] = np.array([])
                        continue

                    t_break = error[-1, 0]
                    track = track[track[:, 0] <= t_break]

            error[:,0] -= error[0, 0]

            errors_dict[i] = error
            tracks_dict[i] = track

        # compute average feature age
        average_age = np.mean([track[-1, 0] - track[0, 0] if len(track) != 0 else 0 for track in tracks_dict.values()])

        # stack errors
        error_stack = np.concatenate([errors for errors in errors_dict.values()], 0)
        error_stack = error_stack[error_stack[:,0].argsort()]

        t_errors = error_stack[:,0]
        euclidean_errors = np.linalg.norm(error_stack[:,1:], axis=1)

        average_error = np.mean(euclidean_errors)

        t_max = np.array(sorted([errors[-1,0] for errors in errors_dict.values()]))
        perc = np.arange(1, 0, -1.0/len(errors_dict))

        # find percentages at t_errors
        euclidean_errors, t_mean = mean_filtering(euclidean_errors, t_errors)

        # interpolate percentage at t_errors
        perc_interp = np.interp(t_mean, t_max, perc)
        perc_interp[0] = 1

        ax.fill_between(t_mean, perc_interp + euclidean_errors, euclidean_errors - perc_interp,
                        color=color, alpha=0.5)
        ax.plot(t_mean, euclidean_errors, color=color, label=label.replace("_", " "), linewidth=2)

        dataset_name = os.path.basename(files[0][0])
        if label not in summary:
            summary[label] = {}
        if dataset_name not in summary[label]:
            summary[label][dataset_name] = {}

        # generate summary
        m_age = round(average_age, 2)
        m_tot = round(average_error, 2)

        summary[label][dataset_name]["feature_age"] = "%s" % m_age
        summary[label][dataset_name]["tracking_error"] = "%s" % m_tot

    ax.set_ylim([0, error_threshold])
    ax.set_xlim(left=0)

    ax.xaxis.set_tick_params(width=3, length=7, labelsize=20, pad=10)
    ax.yaxis.set_tick_params(width=3, length=7, labelsize=20, pad=10)

    fontProperties = {"weight": "bold"}
    x_ticks = [t for t in ax.get_xticks()]
    y_ticks = [t for t in ax.get_yticks()]
    ax.set_xticklabels(x_ticks, fontProperties)
    ax.set_yticklabels(y_ticks, fontProperties)

    ax.set_xlabel("Time [s]", fontsize=30)
    ax.set_ylabel("Error [pixels]", fontsize=30)

    handles, ax_labels = ax.get_legend_handles_labels()
    latex_labels = [r"\textbf{%s}" % l for l in ax_labels]
    ax.legend(handles, latex_labels, frameon=False, prop={"weight": "bold", "size": 20}, loc=1)

    return fig, ax, summary

def format_summary(s):
    """
    Summary has form {method: {dataset: {"tracking_error": ..., "feature_age": ...}}}
    Prints summary in form:

             | Dataset1 | Dataset 2 ....
    --------------------------------------------------------
    Method1 | age11 ...
    ...

             | Dataset1 | Dataset 2 ....
    --------------------------------------------------------
    Method1 | error11 ...
    ...

    """
    import tabulate
    assert len(s) > 0

    methods = list(s.keys())
    header = list(s[methods[0]].keys())
    e_data = [[m] + [s[m][h]["tracking_error"] for h in header] for m in methods]
    t_data = [[m] + [s[m][h]["feature_age"] for h in header] for m in methods]

    age_table = tabulate.tabulate(t_data, headers=header, tablefmt="orgtbl")
    error_table = tabulate.tabulate(e_data, headers=header, tablefmt="orgtbl")

    return error_table, age_table