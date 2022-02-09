import math
from sys import exit
import argparse
import itertools
import os
import csv
from collections import namedtuple
from pprint import pprint

import tensorflow as tf
import numpy as np
from scipy.optimize import root_scalar


class CrossEntropyStoppingCallback(tf.keras.callbacks.Callback):

    def __init__(self, ce_level=0.1):
        super(CrossEntropyStoppingCallback, self).__init__()
        self.ce_level = ce_level

    def on_epoch_end(self, epoch, logs=None):
        if logs["loss"] <= self.ce_level:
            print(f"Stopping as cross entropy is below {self.ce_level}")
            self.model.stop_training = True

        if tf.math.is_nan(logs["loss"]):
            print(f"\n\n\n\n Stopping as NaN values\n\n\n")
            self.model.stop_training = True
            exit()



class DenseLayer(tf.keras.layers.Dense):
    " Version of keras dense layer with perturbable weights."

    def store_weights(self):
        self.stored_weights = tf.Variable(self.weights[0], trainable=False)

    def call_with_noise(self, input_tensor, noise_sigma):
        x = tf.matmul(input_tensor, self.kernel)
        norms = tf.norm(x, axis=-1, keepdims=True)
        noise = tf.random.normal(x.shape, stddev=(norms * noise_sigma))
        return self.activation(x + noise)


class ErfModel(tf.keras.Model):
    "One hidden layer with (normalised) erf activation."

    def __init__(self, width, shape=(28, 28)):
        super(ErfModel, self).__init__(name='')
        self.flat = tf.keras.layers.Flatten(input_shape=shape)
        self.erf = DenseLayer(width, activation=erf_activation, use_bias=False)
        self.final = DenseLayer(10, use_bias=False)

    def call(self, input_tensor, training=False):
        x = self.flat(input_tensor)
        x = self.erf(x)
        x = self.final(x)
        return x

    def store_weights(self):
        for l in self.layers[1:]:
            l.store_weights()

    def call_with_noise(self, input_tensor, noise_sigma=0.):
        # No noise!
        return self.call(input_tensor)

    def store_weights(self):
        for l in self.layers[1:]:
            l.store_weights()
        self.erf.store_weights()

    def kl_extra_sigma_1(self):
        return 0.    # This is zero


class PartialAggModel(tf.keras.Model):
    "Erf final layer with 3 layers of randomised weights."

    def __init__(self, erf_width, relu_width, shape=(28, 28)):
        super(PartialAggModel, self).__init__(name='')
        self.flat = tf.keras.layers.Flatten(input_shape=shape)
        self.h1 = DenseLayer(relu_width, activation="relu", use_bias=False)
        self.h2 = DenseLayer(relu_width, activation="relu", use_bias=False)
        self.h3 = DenseLayer(relu_width, activation="relu", use_bias=False)
        self.erf = DenseLayer(erf_width, activation=erf_activation, use_bias=False)
        self.final = DenseLayer(10, use_bias=False)

    def call(self, input_tensor, training=False):
        x = self.flat(input_tensor)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h3(x)
        x = self.erf(x)
        x = self.final(x)
        return x

    def call_with_noise(self, input_tensor, noise_sigma, training=False):
        x = self.flat(input_tensor)
        x = self.h1.call_with_noise(x, noise_sigma)
        x = self.h2.call_with_noise(x, noise_sigma)
        x = self.h3.call_with_noise(x, noise_sigma)
        x = self.erf(x)
        x = self.final(x)
        return x

    def store_weights(self):
        for l in self.layers[1:]:
            l.store_weights()
        self.erf.store_weights()

    def kl_extra_sigma_1(self):
        h1 = tf.norm(self.h1.stored_weights - self.h1.weights[0]) ** 2
        h2 = tf.norm(self.h2.stored_weights - self.h2.weights[0]) ** 2
        h3 = tf.norm(self.h3.stored_weights - self.h3.weights[0]) ** 2
        return 0.5 * (h1 + h2 + h3)




def erf_activation(x):
    "Erf activation function (with rescaling by norm and sqrt 2)."
    norm = tf.norm(x, keepdims=True, axis=-1)
    return tf.math.erf(x / norm / tf.sqrt(2.))


def get_margins(model, data, stddev=0.):
    "Return an array of the margins of each data point."
    images, labels = data
    preds = model.call_with_noise(images, stddev).numpy()
    yth_score = preds[np.arange(len(preds)), labels]    # f(x)[y] for each x.
    preds[np.arange(len(preds)), labels] = -np.inf      # set these to -inf
    max_not_y = tf.reduce_max(preds, axis=-1)
    margins = yth_score - max_not_y
    # class_error_from_margin = tf.reduce_mean(tf.cast(margins > 0, tf.float32))
    # print("Classification Error from Margins", class_error_from_margin)
    return margins


def margin_error_at(margin, data_margins):
    return tf.reduce_mean(tf.cast(data_margins < margin, tf.float32))


def plot_margins(margins):
    import matplotlib.pyplot as plt
    values, base = np.histogram(margins.numpy(), bins=30)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c='blue')
    plt.grid()
    plt.show()


def bound_complexity_term(model, m_prime, alpha=1.00001):
    "Bound complexity term at margin=1"
    initial_hidden_weights = model.erf.stored_weights
    final_hidden_weights = model.erf.weights[0]
    final_output_weights = model.final.weights[0]
    v_infty = tf.reduce_max(final_output_weights)

    output_term = (tf.norm(final_output_weights) ** 2) * np.log(2)
    hidden_norm = tf.norm(final_hidden_weights - initial_hidden_weights)
    hidden_term = 0.5 * ((v_infty * hidden_norm) ** 2)
    norm_margin = (alpha ** 2) * model.erf.units
    return 17 * norm_margin * (output_term + hidden_term) * np.log(m_prime)


def bound_rhs(model, gamma, m_prime, n_data, delta=0.01, alpha=1.00001):
    "Return the right hand side of the bound in paper."
    n_hidden = model.erf.units
    v_infty = tf.reduce_max(model.final.weights[0])
    cover = (alpha ** 2) * v_infty * n_hidden / gamma
    rhs = 2 * np.log(np.log(cover) / np.log(alpha))
    rhs += np.log(4 * np.sqrt(n_data) / delta)
    rhs += bound_complexity_term(model, m_prime, alpha) / (gamma ** 2)
    return rhs / n_data


def small_kl(q, p):
    return q * math.log(q/p) + (1 - q) * math.log((1 - q) / (1 - p))


def invert_small_kl(train_loss, rhs):
    "Get the inverted small-kl, largest p such that kl(train : p) \le rhs"
    start = train_loss + np.sqrt(rhs / 2.)    # start at McAllester
    try:
        res = root_scalar(lambda r: small_kl(train_loss, r) - rhs,
                        x0=start,
                        bracket=[train_loss, 1. - 1e-20])
    except ValueError:
        return 1.
    return res.root


def calc_bound(model, n_data, train_margins, rhs_extra):
    "Calculate the optimised (over margins) value of the bound."
    possible_margins = np.arange(0.1, 8, 0.1)
    possible_m_prime = [10., 25., 50., 100., 200.]
    margin_bounds = []
    for (gamma, m_prime) in itertools.product(possible_margins, possible_m_prime):
        rhs = bound_rhs(model, gamma, m_prime, n_data)
        rhs += rhs_extra
        err = margin_error_at(gamma, train_margins) + (1/m_prime)
        bound = invert_small_kl(err, rhs) + (1/m_prime)
        margin_bounds += [(gamma, bound, rhs, err)]
    min_bound = min(margin_bounds, key=lambda t: t[1])
    # print(f"Minimum bound of {min_bound[1]} at margin {min_bound[0]}")
    # print(f"Right Hand Side: {min_bound[2]}, Margin error: {min_bound[3]}")
    return min_bound


def load_data(train_size, cifar=False):
    dataset = tf.keras.datasets.cifar10 if cifar else tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # Rescale Data.
    train_images = train_images[:train_size] / 255.0
    train_labels = train_labels[:train_size]
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def big_o_complexity(model, n_data, margin):
    n_hidden = model.erf.units
    initial_hidden_weights = model.erf.stored_weights
    final_hidden_weights = model.erf.weights[0]
    final_output_weights = model.final.weights[0]
    v_infty = tf.reduce_max(final_output_weights)

    output_norm = tf.norm(final_output_weights)
    hidden_norm = tf.norm(final_hidden_weights - initial_hidden_weights)
    final = np.sqrt(n_hidden / n_data) * (output_norm + v_infty * hidden_norm)
    return final / margin


def min_bound_given_sigma(model, sigma, train_data):
    rhs_extra = model.kl_extra_sigma_1() / (sigma ** 2)
    train_margins = get_margins(model, train_data, stddev=sigma)
    return calc_bound(model, len(train_data[0]), train_margins, rhs_extra)


def run(hparams):

    # Load Data.
    (train_images, train_labels), (test_images, test_labels) = load_data(
        hparams["train_size"], cifar=hparams["cifar"])

    # Model setup.
    shape = (28, 28, 3) if hparams["cifar"] else (28, 28)
    if hparams["partial_agg"]:
        no_relus = 100
        model = PartialAggModel(hparams["n_hidden"], no_relus, shape=shape)
    else:
        model = ErfModel(hparams["n_hidden"], shape=shape)

    # Compile and train.
    optimizer = tf.keras.optimizers.SGD(hparams["lr"], momentum=0.9)
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.predict(train_images[0:10])   # ensure layers are built with correct sizes.

    if hparams["pretrain"]:
        pretrain_images = train_images[:hparams["train_size"] // 2]
        pretrain_labels = train_labels[:hparams["train_size"] // 2]
        train_images = train_images[hparams["train_size"] // 2:]
        train_labels = train_labels[hparams["train_size"] // 2:]
        model.fit(pretrain_images, pretrain_labels, epochs=500, batch_size=hparams["batch_size"],
                  callbacks=CrossEntropyStoppingCallback(0.5))

    model.store_weights()

    ce_stopping_condition = 0.3
    model.fit(train_images, train_labels, epochs=2000, batch_size=hparams["batch_size"],
                  callbacks=CrossEntropyStoppingCallback(ce_stopping_condition))

    # Evaluate
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    train_margins = get_margins(model, (train_images, train_labels), stddev=0.)
    train_acc = tf.reduce_mean(tf.cast(train_margins >= 0., tf.float32)).numpy()

    # Best bound.
    bounds = [min_bound_given_sigma(model, sigma, (train_images, train_labels))
              for sigma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]
    min_bound = min(bounds, key=lambda x: x[1])

    # Big O with fixed KL from other layers.
    train_size = hparams["train_size"] // 2 if hparams["pretrain"] else hparams["train_size"]
    sigma = tf.sqrt(model.kl_extra_sigma_1() / tf.cast(train_size, tf.float32))
    train_margins = get_margins(model, (train_images, train_labels), stddev=sigma)
    margin_20pc_loss = np.sort(train_margins)[train_size // 5]
    complexity = big_o_complexity(model, train_size, margin_20pc_loss)

    out_info = {
        "LR": hparams["lr"],
        "N Hidden": hparams["n_hidden"],
        "Train Size": hparams["train_size"],
        "Train accuracy": train_acc,
        "Test accuracy": test_acc,
        "Generalisation Error": train_acc - test_acc,
        "Min bound": min_bound[1],
        "Min bound margin": min_bound[0],
        "Big O": complexity.numpy(),
        "20pc margin": margin_20pc_loss,
    }
    print("\n"*3, "="*20, "Output info", "="*20)
    pprint(out_info)

    filename = "result.csv"
    file_empty = os.stat(filename).st_size == 0

    with open("result.csv", "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_info.keys())
        if file_empty:
            writer.writeheader()
        writer.writerow(out_info)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--train_size", default=60000, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--pretrain", default=False, type=bool)
    parser.add_argument("--partial_agg", default=False, type=bool)
    parser.add_argument("--cifar", default=False, type=bool)
    args = parser.parse_args()

    hparams = {
        "lr": args.lr,
        "n_hidden": args.width,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "partial_agg": args.partial_agg,
        "pretrain": args.pretrain,
        "cifar": args.cifar
    }

    run(hparams)

