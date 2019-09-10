"""
File that takes dataset.yaml path and generates an object containing images and timestamps
"""
import yaml
import numpy as np
import argparse
import os
from os.path import isfile, join, dirname, abspath

from big_pun.tracker import Tracker
from big_pun.tracker_init import TrackerInitializer
from big_pun.tracker_utils import getTrackData, compareTracks
from big_pun.dataset import Dataset

from plotting.plot_tracks import plot_tracks_3d, plot_num_features, format_summary

from feature_track_visualizer.visualizer import FeatureTracksVisualizer


parser = argparse.ArgumentParser(description='''Generates ground truth tracks for a given set of feature tracks. 
The user needs to specify the root where the rosbag containing the images/poses + depth maps are found.
 Additionally, a configuration file for the dataset and tracker must be provided.''')
parser.add_argument('--tracker_params', help='Params yaml-file for tracker algorithm.', default="")
parser.add_argument('--tracker_type', help='Tracker type. Can be one of [reprojection, KLT].', default="")
parser.add_argument('--root', help='Directory where datasets are found.', default="", required=True)
parser.add_argument('--dataset', help="Params yaml-file for dataset.", default="", required=True)
parser.add_argument('--file', help="Tracks file for ground truth computation.", default="", required=True)

parser.add_argument('--plot_3d', help="Whether to do a 3d plot.", action="store_true", default=False)
parser.add_argument('--plot_errors', help="Tracks file giving KLT initialization.", action="store_true", default=False)
parser.add_argument('--error_threshold', help="Error threshold. Tracks which exceed this threshold are discarded.", type=float, default=10)
parser.add_argument('--video_preview', help="Whether to create a video preview.", action="store_true", default=False)


args = parser.parse_args()

if args.tracker_params == "":

    # check what the type is
    assert args.tracker_type != "", "Either tracker_type or tracker_params need to be given."
    assert args.tracker_type in ["KLT", "reprojection"], 'Tracker type must be one of [reprojection, KLT].'
    config_dir = join(dirname(abspath(__file__)), "config")
    args.tracker_params = join(config_dir, "%s_params.yaml" % args.tracker_type)
else:
    assert isfile(args.tracker_params), "Tracker params do not exist."
    assert args.tracker_params.endswith(".yaml") or args.tracker_params.endswith(".yml"), \
        "Tracker params '%s 'must be a yaml file." % args.tracker_params

with open(args.tracker_params, "r") as f:
    tracker_config = yaml.load(f, Loader=yaml.Loader)

assert os.path.isfile(args.file), "Tracks file '%s' does not exist." % args.file
assert "type" in tracker_config, "Tracker parameters must contain a type tag, which can be one of [reprojection, KLT]."
tracker_init_config = {
    "tracks_csv": args.file,
    "type": "tracks" if tracker_config["type"] == "KLT" else "depth_from_tracks"
}

print("Evaluating ground truth for %s in folder %s." % (os.path.basename(args.file), os.path.dirname(args.file)))

# make feature init
tracks_init = TrackerInitializer(args.root, args.dataset, config=tracker_init_config)

# init tracker
tracker = Tracker(tracker_config)

# get tracks
tracked_features = tracker.track(tracks_init.tracks_obj, tracks_init.tracker_params)

# save tracks
out_csv = args.file + ".gt.txt"
print("Saving ground truth files to %s" % os.path.basename(out_csv))
np.savetxt(out_csv, tracked_features, fmt=["%i", "%.8f", "%.4f", "%.4f"])

# load both gt and normal tracks
print("Computing errors")
est_tracks_data = getTrackData(args.file, filter_too_short=True)
gt_tracks_data = getTrackData(out_csv)

# compute errors
error_data = compareTracks(est_tracks_data, gt_tracks_data)
errors_csv = args.file + ".errors.txt"
print("Saving error files to %s" % os.path.basename(errors_csv))
np.savetxt(errors_csv, error_data, fmt=["%i", "%.8f", "%.4f", "%.4f"])

# plotting
folder = os.path.dirname(args.file)
results_folder = os.path.join(folder, "results")
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

if args.plot_3d:
    # create 3D plot of tracks (only first features) and with gt
    print("Saving 3d space-time curves to results/3d_plot.pdf and results/3d_plot_with_gt.pdf. This may take some time.")
    dataset = Dataset(args.root, args.dataset, dataset_type="frames")
    fig1, ax1 = plot_tracks_3d(dataset, args.file, out_csv)
    fig1.savefig(join(results_folder, "3d_plot_with_gt.pdf"), bbox_inches="tight")

    dataset = Dataset(args.root, args.dataset, dataset_type="frames")
    fig2, ax2 = plot_tracks_3d(dataset, args.file)
    fig2.savefig(os.path.join(results_folder, "3d_plot.pdf"), bbox_inches="tight")

if args.video_preview:
    # preview of video
    dataset = Dataset(args.root, args.dataset, dataset_type="frames")
    out_video = os.path.join(results_folder, "preview.avi")

    video_params = {
        'track_history_length': 0.4,
        'scale': 4.0,
        'framerate': 80,
        'speed': 1.0,
        'marker': "circle",
        "error_threshold": args.error_threshold,
        "contrast_brightness": [1, 0],
        "crop_to_predictions": True,
    }

    # load dataset
    print("Saving video preview to results/preview.avi.")
    viz = FeatureTracksVisualizer(file=args.file, dataset=dataset, params=video_params)
    viz.writeToVideoFile(out_video)

if args.plot_errors:
    print("Saving error plot to errors.pdf")
    fig3, ax3, summary = plot_num_features(args.file, error_threshold=args.error_threshold)
    fig3.savefig(os.path.join(results_folder, "errors.pdf"), bbox_inches="tight")

    # write summary into a text file
    print("Saving error and age table in locations results/errors.txt and results/feature_age.txt")
    error_table, age_table = format_summary(summary)
    with open(os.path.join(results_folder, "errors.txt"), "w") as f:
        f.write(error_table)
    with open(os.path.join(results_folder, "feature_age.txt"), "w") as f:
        f.write(age_table)

