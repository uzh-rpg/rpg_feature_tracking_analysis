"""
File that takes dataset.yaml path and generates an object containing images and timestamps
"""
import argparse
import os
from os.path import isdir, isfile

from plotting.plot_tracks import plot_num_features, format_summary


parser = argparse.ArgumentParser(description='''Script that generates a plot and average age/error files for several methods.
                                                The user must supply the root directory where the methods are stored, a config file
                                                and an error threshold.''')

parser.add_argument('--results_directory', help='Directory where results will be stored.', default="")
parser.add_argument('--root', help='Directory where tracks_directories are found', default="", required=True)
parser.add_argument('--config', help="Config file with label and colors for each method.", default="", required=True)

parser.add_argument('--error_threshold', help="All tracks with an error higher than this threshold will be discarded."
                                              "error_threshold < 0 will not discard any tracks.",
                    type=float, default=10)

args = parser.parse_args()

# check that directories and config exist
assert isdir(args.root), "Root directory '%s' is not a directory." % args.root
args.results_directory = args.results_directory or args.root
assert isdir(args.results_directory), "Results directory '%s' is not a directory." % args.results_directory
assert isfile(args.config), "Config file '%s' is not a file." % args.config
assert args.config.endswith(".yaml") or args.config.endswith(".yml"), "Config file '%s' is not a yaml file." % args.config


print("Plotting errors for methods in '%s'" % args.root)
print("Will save all data in %s." % args.results_directory)
fig3, ax3, summary = plot_num_features(f="", root=args.root,
                                       config_path=args.config,
                                       error_threshold=args.error_threshold)

base_dir = os.path.basename(args.results_directory)
print("Saving error plot in location %s/errors.pdf" % base_dir)
fig3.savefig(os.path.join(args.results_directory, "errors.pdf"), bbox_inches="tight")

# write summary into a text file
print("Saving error and age table in locations %s/errors.txt and %s/feature_age.txt" % (base_dir, base_dir))
error_table, age_table = format_summary(summary)
with open(os.path.join(args.results_directory, "errors.txt"), "w") as f:
    f.write(error_table)
with open(os.path.join(args.results_directory, "feature_age.txt"), "w") as f:
    f.write(age_table)
