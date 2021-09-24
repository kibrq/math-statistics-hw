#!/usr/bin/env python3

from typing import Tuple
from bisect import bisect_right


def simulate(windows: int, queue: int, window_settings: Tuple[callable, callable]) -> float:
    serve_time, availabiliy = window_settings
   
    window_availabilities = [availabiliy(i) for i in range(windows)]
    window_times = list(sorted(map(lambda i: (serve_time(i), i), range(windows))))

    result = 0
    for _ in range(queue):
        time, window = 0, 0
        while True:
            if len(window_times) <= 0:
                return 0
            time, window = window_times.pop(0)
            window_availabilities[window] -= 1
            if window_availabilities[window] > 0:
                break
        new_serve = (time + serve_time(window), window)
        place_to_insert = bisect_right(window_times, new_serve)
        window_times.insert(place_to_insert, new_serve)
        result += new_serve[0]
    return result / queue


import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt


generator = r.default_rng(seed = 42)


def exponential(alpha):
    return generator.exponential(alpha, 1)[0]


def serve_distribution(alphas):
    return lambda i: exponential(alphas[i])


def experiment(x_axis, configs):
    values = [(x_axis(c), get_result_from_config(c)) for c in configs]
    xs, ys = zip(*values)
    return xs, ys


def get_result_from_config(config):
    N = 100
    n, queue, alpha, m = config
    return sum([simulate(n, queue, (serve_distribution([alpha] * n), lambda i: m)) for _ in range(N)]) / N


def queue_length():
    n, alpha, m = 100, 1, 2
    xs, ys = experiment(lambda c: c[1], [(n, queue, alpha, m) for queue in range(1, 100)])
    plt.plot(xs, ys, label='m=2')
    n, alpha, m = 100, 1, 10 ** 9
    xs, ys = experiment(lambda c: c[1], [(n, queue, alpha, m) for queue in range(1, 200)])
    plt.plot(xs, ys, label='m=inf')
    plt.legend(loc='lower right')
    plt.title('Queue length (windows=100, alpha=1)')
    plt.show()


def windows_count():
    queue, alpha, m = 100, 1, 2
    xs, ys = experiment(lambda c: c[0], [(n, queue, alpha, m) for n in range(100, 300)])
    plt.plot(xs, ys, label='m=2')
    queue, alpha, m = 100, 1, 10 ** 9
    xs, ys = experiment(lambda c: c[0], [(n, queue, alpha, m) for n in range(50, 300)])
    plt.plot(xs, ys, label='m=inf')
    plt.legend(loc='lower right')
    plt.title('Windows count (queue=100, alpha=1)')
    plt.show()


def alpha():
    n, queue, m = 100, 100, 2
    xs, ys = experiment(lambda c: c[2], [(n, queue, alpha, m) for alpha in np.linspace(0.5, 1.5, 100)])
    plt.plot(xs, ys, label='m=2')
    n, queue, m = 100, 100, 10 ** 9
    xs, ys = experiment(lambda c: c[2], [(n, queue, alpha, m) for alpha in np.linspace(0.5, 1.5, 100)])
    plt.plot(xs, ys, label='m=inf')
    plt.legend(loc='lower right')
    plt.title('Alpha (windows=100, queue=100)')
    plt.show()


from argparse import ArgumentParser
from sys import argv


def main(args):
    parser = ArgumentParser()
    parser.add_argument('parameter', choices=['queue-length', 'windows-count', 'alpha'])
    param = parser.parse_args(args).parameter
    if param == 'queue-length':
        queue_length()
    elif param == 'windows-count':
        windows_count()
    else:
        alpha()


if __name__ == '__main__':
    main(argv[1:])
