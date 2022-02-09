import csv
from statistics import mean, stdev, median
import matplotlib.pyplot as plt
from itertools import product, combinations
import numpy as np
import seaborn as sns


def rename_key(dic, old_key, new_key):
    "Rename a dictionary key."
    dic[new_key] = dic.pop(old_key)


def sign(x):
    return 1. if x >= 0. else -1.


def filter_results(list_of_dicts, search):
    """Filter list to those with bits matching search.

    e.g. filter_results([...], {"foo" = "bar",})
        => [{"foo" = "bar", "x" = 1}, {"foo" = "bar", "x" = 2}]
    """
    # Check for containment with dict1.items() >= dict2.items().
    return list(filter(lambda d: d.items() >= search.items(), list_of_dicts))


def group_by_hparam(list_of_dicts, x_label, xs):
    "Returns a list (sorted by xs) of list of dicts with same xs."
    return [filter_results(list_of_dicts, {x_label: x}) for x in xs]


def means_list_list_dict(ll_dicts, key):
    "Returns mean value of key from of sub-lists of dicts."
    return [mean(d[key] for d in ds) for ds in ll_dicts]


def stdev_list_list_dict(ll_dicts, key):
    "Returns mean value of key from of sub-lists of dicts."
    return [stdev(d[key] for d in ds) for ds in ll_dicts]


def pretty_dict_format(dic):
    return "  ".join([f"{k}: {v}" for k, v in dic.items()])


def plot_double(ax, results, filters, xlabel, xs, gen_range=0.05):
    "Plot Complexity and Generalisation vs xlabel for xs after applying filters."
    filtered = filter_results(results, filters)
    by_x = group_by_hparam(filtered, xlabel, xs)

    comp_means = means_list_list_dict(by_x, "Big O")
    comp_stdevs = stdev_list_list_dict(by_x, "Big O")
    gen_means = means_list_list_dict(by_x, "Generalisation Error")
    gen_stdevs = stdev_list_list_dict(by_x, "Generalisation Error")

    error_kw = {"capsize": 5, "capthick": 1}
    fontsize_axes = 12
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_axes)
    ax.set_title(pretty_dict_format(filters), fontsize=14)
    ax.tick_params(axis="x", which="minor")
    # ax.set_xticks([10**4, 6*10**4])
    # ax.set_xticklabels([10**4, 6*10**4])



    ax.set_ylabel("Complexity Measure", fontsize=fontsize_axes)
    comp = ax.errorbar(
        xs,
        comp_means,
        color="red",
        yerr=comp_stdevs,
        fmt=":o",
        label="Complexity",
        **error_kw)
    ax.set_ylim(0., 5.)

    right_ax = ax.twinx()
    right_ax.set_ylabel("Generalisation Error", fontsize=fontsize_axes)
    right_ax.set_ylim(0., gen_range)
    gen = right_ax.errorbar(
        xs,
        gen_means,
        color="green",
        yerr=gen_stdevs,
        fmt="--o",
        label="Generalisation Error",
        **error_kw)
    ax.legend([comp, gen], [comp.get_label(), gen.get_label()])


def get_results(filename):
    "Load the CSV file."
    with open(filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        results = [{k: float(v) for k, v in row.items()} for row in reader]

    # Rename N Hidden to Width
    for d in results:
        rename_key(d, "N Hidden", "Width")

    return results


def lr_plot(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.tight_layout(pad=5.)
    fig.suptitle("Generalisation and Complexity Comparison by Learning Rate", fontsize=16)
    lrs = [0.1, 0.03, 0.01, 0.003, 0.001]
    plot_double(ax1, results, {"Width": 100, "Train Size": 60000}, "LR", lrs)
    plot_double(ax2, results, {"Width": 100, "Train Size": 30000}, "LR", lrs)
    plot_double(ax3, results, {"Width": 200, "Train Size": 60000}, "LR", lrs)
    fig.savefig("Figure_lr")


def width_plot(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.tight_layout(pad=5.)
    fig.suptitle("Generalisation and Complexity Comparison by Width", fontsize=16)
    widths = [50, 100, 200, 400, 800]
    plot_double(ax1, results, {"LR": 0.01, "Train Size": 60000}, "Width", widths)
    plot_double(ax2, results, {"LR": 0.01, "Train Size": 30000}, "Width", widths)
    plot_double(ax3, results, {"LR": 0.001, "Train Size": 60000}, "Width", widths)
    fig.savefig("Figure_width")


def tsize_plot(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.tight_layout(pad=5.)
    fig.suptitle("Generalisation and Complexity Comparison by Training Size", fontsize=16)
    tsizes = [60000, 30000, 15000, 7500]
    plot_double(ax1, results, {"LR": 0.01, "Width": 200}, "Train Size", tsizes)
    plot_double(ax2, results, {"LR": 0.01, "Width": 100}, "Train Size", tsizes)
    plot_double(ax3, results, {"LR": 0.001, "Width": 200}, "Train Size", tsizes)
    fig.savefig("Figure_tsize")


def lr_plot_p(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.tight_layout(pad=5.)
    fig.suptitle("Generalisation and Complexity Comparison by Learning Rate", fontsize=16)
    lrs = [0.01, 0.003, 0.001]
    plot_double(ax1, results, {"Width": 100, "Train Size": 60000}, "LR", lrs)
    plot_double(ax2, results, {"Width": 100, "Train Size": 30000}, "LR", lrs)
    plot_double(ax3, results, {"Width": 200, "Train Size": 60000}, "LR", lrs)
    fig.savefig("Figure_lr_p")


def width_plot_p(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.tight_layout(pad=5.)
    fig.suptitle("Generalisation and Complexity Comparison by Width", fontsize=16)
    widths = [50, 100, 200, 400, 800]
    plot_double(ax1, results, {"LR": 0.01, "Train Size": 60000}, "Width", widths)
    plot_double(ax2, results, {"LR": 0.01, "Train Size": 30000}, "Width", widths)
    plot_double(ax3, results, {"LR": 0.001, "Train Size": 60000}, "Width", widths)
    fig.savefig("Figure_width_p")


def tsize_plot_p(results):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(12, 5, forward=True)
    fig.tight_layout(pad=5.)
    fig.suptitle("Generalisation and Complexity Comparison by Training Size", fontsize=16)
    tsizes = [60000, 30000, 15000, 7500]
    plot_double(ax1, results, {"LR": 0.01, "Width": 200}, "Train Size", tsizes, gen_range=0.06)
    plot_double(ax2, results, {"LR": 0.01, "Width": 100}, "Train Size", tsizes, gen_range=0.06)
    plot_double(ax3, results, {"LR": 0.001, "Width": 200}, "Train Size", tsizes, gen_range=0.06)
    fig.savefig("Figure_tsize_p")


def check_results_exist(results, lrs, widths, tsizes):
    for l, w, s in product(lrs, widths, tsizes):
        filt = {"LR": l, "Width": w, "Train Size": s}
        if len(filter_results(results, filt)) == 0:
            print("No entry for:", filt)
        if len(filter_results(results, filt)) == 1:
            print("Only one entry for:", filt)



def sign_error(r1, r2):
    g1 = r1["Generalisation Error"]
    g2 = r2["Generalisation Error"]
    c1 = r1["Big O"]
    c2 = r2["Big O"]
    return 0.5 * (1 - sign(g2 - g1) * sign(c2 - c1))

def mean_sign_error(rs1, rs2):
    "Return expected sign error from sets of results with different hparams."
    return mean([sign_error(r1, r2) for r1, r2 in zip(rs1, rs2)])


def sign_errors_experiments(experiments):
    sign_errors = []
    for sign_experiment in experiments:
        results1 = filter_results(results, sign_experiment[0])
        results2 = filter_results(results, sign_experiment[1])
        sign_errors.append(mean_sign_error(results1, results2))
    return sign_errors



def eval_sign_errs(experiments):
    sgn_errors = sign_errors_experiments(experiments)
    print("Max Sign Error = ", max(sgn_errors),
        ", Median Sign Error = ", median(sgn_errors),
          " Average Sign Error = ", mean(sgn_errors), "\n\n")


def sign_error_evaluations(results, lrs, widths, tsizes):
    experiments_tsize = [
        ({"LR": lr, "Width": wid, "Train Size": t1},
        {"LR": lr, "Width": wid, "Train Size": t2})
        for t1, t2 in combinations(tsizes, 2) for lr, wid in product(lrs, widths)]

    experiments_widths = [
        ({"LR": lr, "Width": w1, "Train Size": ts},
        {"LR": lr, "Width": w2, "Train Size": ts})
        for w1, w2 in combinations(widths, 2) for lr, ts in product(lrs, tsizes)]

    experiments_lrs = [
        ({"LR": lr1, "Width": wid, "Train Size": ts},
        {"LR": lr2, "Width": wid, "Train Size": ts})
        for lr1, lr2 in combinations(lrs, 2) for wid, ts in product(widths, tsizes)]

    print("Learning Rate Experiments:")
    eval_sign_errs(experiments_lrs)

    print("Width Experiments:")
    eval_sign_errs(experiments_widths)

    print("Train Size Experiments:")
    eval_sign_errs(experiments_tsize)

    print("All Experiments:")
    eval_sign_errs(experiments_tsize + experiments_lrs + experiments_widths)




results = get_results("result_erf.csv")

lrs = [0.1, 0.03, 0.01, 0.003, 0.001]
widths = [50, 100, 200, 400, 800]
tsizes = [60000, 30000, 15000, 7500]
check_results_exist(results, lrs, widths, tsizes)

lr_plot(results)
width_plot(results)
tsize_plot(results)

sign_error_evaluations(results, lrs, widths, tsizes)




results = get_results("result_partial.csv")

lrs = [0.01, 0.003, 0.001]
widths = [50, 100, 200, 400, 800]
tsizes = [60000, 30000, 15000, 7500]
check_results_exist(results, lrs, widths, tsizes)
sign_error_evaluations(results, lrs, widths, tsizes)

lr_plot_p(results)
width_plot_p(results)
tsize_plot_p(results)

plt.show(block=False)
input("hit[enter] to end.")
plt.close('all')

