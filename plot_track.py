import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm


parser = argparse.ArgumentParser("")
parser.add_argument("--file",  default="")
parser.add_argument("--id", type=int, default=-1)

args = parser.parse_args()

assert os.path.isfile(args.file), "Tracks file must exist."

data = np.genfromtxt(args.file)
gt = np.genfromtxt(args.file+".gt.txt")
errors = np.genfromtxt(args.file+".errors.txt")

ids = np.unique(data[:,0]).astype(int)
if args.id != -1:
    assert args.id in ids
    ids = [args.id]

folder = os.path.dirname(args.file)

results_folder = os.path.join(folder, "results")
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)
tracks_folder = os.path.join(results_folder, "tracks")
if not os.path.isdir(tracks_folder):
    os.mkdir(tracks_folder)

for i in tqdm.tqdm(ids):

    # get one track
    est_one_track = data[data[:, 0] == i, 1:]
    gt_one_track = gt[gt[:, 0] == i,  1:]
    errors_one_track  = errors[errors[:, 0] == i, 1:]

    fig, ax = plt.subplots(nrows=3)
    t0 = est_one_track[0,0]

    # plot x coordinates
    ax[0].plot(est_one_track[:,0]-t0, est_one_track[:,1], color="b", label="estimate")
    ax[0].plot(gt_one_track[:,0]-t0, gt_one_track[:,1], color="g", label="ground truth")
    ax[0].set_ylabel("X coords. [px]")
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, borderaxespad=0., frameon=False)

    ax[0].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    ax[1].plot(est_one_track[:,0]-t0, est_one_track[:,2], color="b")
    ax[1].plot(gt_one_track[:,0]-t0, gt_one_track[:,2], color="g")
    ax[1].set_ylabel("Y coords. [px]")

    ax[1].tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    ax[2].plot(errors_one_track[:,0]-t0, errors_one_track[:,1], color="c", label="error x")
    ax[2].plot(errors_one_track[:,0]-t0, errors_one_track[:,2], color="r", label="error y")
    ax[2].plot(errors_one_track[:,0]-t0, np.sqrt(errors_one_track[:,2]**2+errors_one_track[:,1]**2), color="m", label="total error")
    ax[2].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 ncol=3, borderaxespad=0., frameon=False)

    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Error [px]")

    plt.subplots_adjust(hspace=0.3)

    fig.savefig(os.path.join(tracks_folder, "track_%s.pdf" % i), bbox_inches="tight")
    plt.close()