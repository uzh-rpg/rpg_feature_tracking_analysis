import argparse
from big_pun.bag2dataframe import Bag2Images
from feature_track_visualizer.visualizer import FeatureTracksVisualizer
import os
from os.path import isfile

example_dir = os.path.join(os.path.dirname(__file__), "example")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Folder containing the tracks files", nargs="+", default=[], required=True)
parser.add_argument("--file", help="Folder containing the tracks files", default="", required=True)

# rendering of track
parser.add_argument("--track_history_length", type=float, help="Maximum track length in seconds", default=0.4)
parser.add_argument("--scale", type=float, help="Scaling factor", default=4.0)
parser.add_argument("--framerate", type=float, help="Framerate (specifies the time between two subsequent frames in the visualization)", default=80.0)
parser.add_argument("--speed", type=float, help="Speed factor", default=1.0)
parser.add_argument('--marker', type=str, default="circle")

parser.add_argument('--error_threshold', type=float, default=-1)
parser.add_argument('--crop_to_predictions', action="store_true", default=False)

# write out
parser.add_argument('--video_file', type=str, default="")

args = parser.parse_args()


assert isfile(args.file), "Tracks file must exist."
assert len(args.dataset) == 2, "--dataset must contain path to rosbag and image topic"
assert isfile(args.dataset[0]), "Bag must exist."
assert args.track_history_length >=0, "--track_history_length must be non-negative."
assert args.scale > 0, "--scale must be positive."
assert args.framerate > 0, "--framerate must be positive."
assert args.speed > 0, "seed must be positive."
assert args.marker in ["circle", "cross"], "--marker must be circle or cross."

params = {}
params['track_history_length'] = args.track_history_length
params['scale'] = args.scale
params['framerate'] = args.framerate
params['speed'] = args.speed
params['marker'] = args.marker

params["crop_to_predictions"] = args.crop_to_predictions
params["error_threshold"] = args.error_threshold

# load dataset
dataset = Bag2Images(args.dataset[0], args.dataset[1])

viz = FeatureTracksVisualizer(file = args.file, dataset = dataset, params = params)
try:
    if args.video_file != "":
        viz.writeToVideoFile(args.video_file)
    else:
        viz.visualizationLoop()
except KeyboardInterrupt:
    viz.cleanup()
