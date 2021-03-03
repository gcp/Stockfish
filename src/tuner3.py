#!/usr/bin/env python3

import logging
import subprocess
import nevergrad as ng
import re
import struct
import os

def write_nnue(bias2, weights2, bias3, weights3):
    ibias2 = [int(x) for x in bias2]
    iweights2 = [int(x) for x in weights2]
    iweights3 = [int(x) for x in weights3]
    print(ibias2, iweights3, bias3)
    with open('nn-tune.nxxx', mode='r+b') as fh:
        fh.seek(21021509)
        fh.write(struct.pack('<32i', *ibias2))
        fh.write(struct.pack('<1024b', *iweights2))
        fh.write(struct.pack('<1i', bias3))
        fh.write(struct.pack('<32b', *iweights3))


def run_test(args, bias2, weights2, bias3, weights3) -> float:
    write_nnue(bias2, weights2, bias3, weights3)
    cmdl = "cutechess-cli "
    cmdl += "-engine cmd=./stockfish "
    for n in range(3):
        cmdl += "option.tune_array[{}]={} ".format(n, args[n])
    cmdl += "option.EvalFile=nn-tune.nxxx "
    cmdl += "-engine cmd=./stockfish_master dir=. "
    cmdl += "-each proto=uci tc=2+0.02 book=book6.bin -pgnout tune.pgn "
    cmdl += "-repeat -games 16 -concurrency 4 -resign movecount=3 score=500 "
    cmdl += "-draw movenumber=40 movecount=10 score=10 -tb ~/egtb "
    print(cmdl)
    proc = subprocess.run([cmdl],
                          shell=True, capture_output=True, text=True)
    prog = re.compile("Score of .+ vs .+: (\d+) - (\d+) - (\d+) .*")
    wld = None
    for line in proc.stdout.splitlines():
        m = prog.match(line)
        if m:
            wld = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    score = (wld[0] + wld[2] * 0.5) / 16
    fitness = 1.0 - score
    print(" ", wld, fitness, flush=True)
    # print("{} fitness={}".format(str(wld), fitness))
    # Minimize fitness, i.e. minimize original engine score
    return fitness


# Instrumentation class is used for functions with multiple inputs
# (positional and/or keywords)
parametrization = ng.p.Instrumentation(
    ng.p.Scalar(lower=214, upper=1282).set_integer_casting(),
    ng.p.Scalar(lower=227, upper=1364).set_integer_casting(),
    ng.p.Scalar(lower=59, upper=352).set_integer_casting(),
    bias2=ng.p.Array(shape=(32,)).set_integer_casting().set_bounds(
        lower=-2**16, upper=2**16),
    weights2=ng.p.Array(shape=(32*32,)).set_integer_casting().set_bounds(lower=-
                                                                         127, upper=127),
    bias3=ng.p.Scalar(lower=-2**16, upper=2**16).set_integer_casting(),
    weights3=ng.p.Array(shape=(32,)).set_integer_casting().set_bounds(
        lower=-127, upper=127),
)

optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=5000)

optimizer.suggest(641, 682, 176,
                  bias2=[-6683, 8069, -4843, -8680, 9799, -11012, -10695, 571, -5559,
                         -5293, -2670, -6079, 1297, -1331, -9148, -10851, -3941, -4830, -2393,
                         -6169, 2991, -2562, -4842, -11015, 5335, -2852, 1087, 2347, -1115,
                         -733, -2040, -5415],
                  weights2=[-5, 15, -3, 6, -8, 3, -11, -11, -16, 21, 51, -8, -27, -8, 6, 1, -1, 23, -4, 14, 23, -2, 23, -11, 9, 21, 1, -1, -5, 1, 11, -7, -5, 35, 41, -14, 5, -29, -1, 8, -56, -8, 5, 19, -5, 0, 9, -7, -12, 13, -40, -9, 31, 42, -122, -15, -45, -18, -64, -7, 25, -12, -14, 1, -22, 0, 19, 42, -1, 18, -9, 9, -21, -1, 7, -96, 14, -23, 9, -11, 22, -55, -20, 27, -5, -4, -31, -26, -14, -13, -101, 33, -2, -24, 2, -27, 44, -13, 37, 31, 8, 15, -37, 20, -30, -9, -12, -32, 30, 20, -7, 0, 5, 10, -1, -14, 49, -9, -5, -21, -7, 20, 47, 14, -19, -19, -7, 1, -11, 18, 35, -15, -12, 22, 0, 7, -17, 37, 15, -16, -31, -22, -8, -7, 11, -26, -37, 36, -35, -30, -37, -14, -47, -10, 4, 0, 25, -30, 9, -32, 21, -52, -26, 36, 23, 4, -76, 29, 7, -40, -106, 14, 32, 2, 18, -17, -6, 28, 1, -11, 23, 5, 21, 31, 23, 3, 3, -22, -88, -16, -25, 18, -48, 27, 20, 57, -2, 20, 28, -6, -42, -14, 9, -77, -76, -48, 6, -33, 44, -103, -70, 17, 1, 1, -15, -6, -11, 10, -27, 27, 5, -42, 8, -43, -4, -7, 6, -42, -3, 2, 3, 4, 0, -14, -12, 69, -34, 25, -9, 4, -14, -32, -5, -19, 6, -13, -13, -6, -16, -4, 2, -6, 10, 2, -4, -1, 13, -15, -13, 0, 34, 8, -65, 29, 5, -23, -22, -3, -5, -4, 28, -4, -16, 15, 13, -18, 20, -40, 29, 26, 56, 3, 8, -20, -48, -15, -56, 4, 19, -21, -15, -7, 22, -1, -49, 16, 5, -18, -26, 5, 2, -1, 13, -4, -25, 18, 3, -11, 16, 5, 20, 17, 30, 12, 9, -21, -31, -16, -30, -1, 6, -15, -4, -19, -18, -12, -55, 8, 10, -16, -18, 11, -28, -7, -4, -6, -8, -41, 10, 3, -7, -15, -29, 20, 54, -17, 8, 36, -43, -29, -44, 6, 1, 17, 23, 12, 17, 10, 1, 4, 3, -12, 0, -1, 34, 12, 8, 39, 38, -28, -1, -29, 47, -14, -16, 29, -48, 6, 1, -7, 4, -18, -5, 7, -5, 1, -18, 32, -8, 5, -11, -14, -22, -5, 3, -11, -24, -21, 49, -8, 6, -69, -10, -27, -15, -19, -1, -3, 8, 0, -53, -2, -11, 19, -18, 42, -20, 20, 18, 30, -4, 12, 12, 5, -23, 14, 0, -37, -57, -26, 0, -11, 13, -51, -29, 31, -14, -14, -29, -5, -19, 8, -42, 13, 6, -24, 5, -40, -1, -24, 24, 4, 19, 74, 12, 11, 1, -15, -15, -3, 0, 21, 0, -9, 22, -2, -7, 1, -11, 53, -30, -1, -10, -2, -68, 11, -30, -8, 27, -1, 9, -9, -18, 6, 32, 8, -103, 26, 21, -27, -32, -9, -27, -1, 37, -1, 10, 0, 8, -13, 3, 11, 16, 10,
                            49, -17, 11, -28, -30, -8, -34, 1, 21, 56, 0, -19, 23, -9, -6, -19, -3, -7, 3, 5, -1, -13, 17, 20, 8, 35, 9, 18, 24, -7, 10, -6, 8, -2, 1, -6, -9, -2, -12, -5, 3, -25, -11, -6, 17, -1, -41, 8, 3, -8, -12, 10, 5, 16, 33, 1, -20, 37, 4, -4, 47, -6, 43, 10, 38, 16, -7, -20, -19, -7, 4, 4, 6, 7, 7, -14, -3, 9, 7, 1, 17, 8, 3, 1, -8, 13, 2, -2, -18, 41, -14, 12, 48, 9, -1, 5, 8, -2, 6, 7, 12, -5, -2, -9, -18, 1, -19, 27, 18, 9, 2, -1, 7, -11, 11, 3, -27, 7, -17, -4, 14, 14, -19, -24, -8, -4, -14, 0, -19, -4, -24, 16, 15, 6, 27, 32, 17, 14, -9, -7, -2, 5, 47, -14, -13, 9, -7, 10, -52, -1, 6, -66, 13, -15, -16, -29, 6, 6, 18, 1, 7, 17, 29, -35, 38, -6, 13, -8, -1, 10, 7, 9, 2, 0, 6, -8, -26, 35, 35, -10, -34, -27, -20, 2, 6, -4, -1, -2, -50, 9, -28, -4, -18, 7, -4, 7, 14, 1, -11, -1, -7, -13, -27, 54, 1, -33, 10, -1, 6, -31, -13, 17, 2, 12, -1, 5, -2, 2, 23, -32, 23, 28, 19, 10, -13, 15, 1, -4, -17, 22, -1, 10, -6, -25, -37, 30, 20, -22, 46, -6, 11, -10, 4, 1, -17, 9, 14, -2, -5, 22, -19, -26, 12, 3, 21, 1, 21, 1, -6, 9, 2, 13, 32, 19, -16, 9, 34, 13, -6, 12, -4, 12, -14, 24, 2, -33, -64, -28, -7, -13, 10, -41, -44, 40, -35, -33, -38, -14, -30, 9, -14, 8, 11, -32, 10, -59, -2, 66, -18, 10, 4, -5, 6, 3, -22, 24, -9, -13, 0, -3, 0, -4, -5, 7, 8, -1, 8, -2, -7, -2, -9, -5, -34, -7, 66, -37, 2, -2, -23, 28, 20, 23, -14, 23, -1, -8, -28, -10, 6, -8, -49, -14, 17, -12, 25, -24, -15, -36, -5, -8, -33, -11, -15, 8, -30, 20, 4, 3, -11, 58, -19, 4, 2, 28, -5, 3, -27, -1, -2, -4, 13, -12, -24, 5, 21, -5, 10, -23, -13, 4, 5, -39, 17, -10, -32, -10, -15, 8, -16, 17, 2, 13, -2, 11, 4, -23, -22, 17, 19, -12, -18, 16, 9, -12, -41, -5, -55, -10, 26, -33, -15, -8, -54, -16, 12, -14, -35, -23, -5, 15, 16, 4, -2, -18, 20, 0, -5, -22, 22, 7, -32, 10, 5, -17, -17, 3, -1, 10, 10, -5, -13, 21, 6, -5, 18, -8, 15, 19, -30, 6, 9, -13, -28, -16, -30, 2, -17, 6, 0, 55, -20, -4, 59, -3, -3, -21, -9, 5, -4, -13, 5, 15, -14, -7, 5, -13, -7, -11, 29, 10, -2, 5, -20, 19, 28, 11, 14, -5, 25, -13, -2, -39, 20, -18, -12, 12, -9, -5, -23, 16, -6, 4, 14, 3, -28, 34, 13, -12, 26, 17, 37, 13, 35, 29, 7, -22, -13, -22, -38, 6],
                  bias3=-193,
                  weights3=[-27, -16, -76, 57, -21, 121, -118, 25, 31, 52, -34, 22, 13,
                            -37, -20, 96, -57, 34, 36, 41, -18, -19, 16, -31, -12, -36,
                            -22, -10, -33, 26, -12, 18])

for n in range(optimizer.budget):
    x = optimizer.ask()
    print("Step={}/{} ".format(n + 1, optimizer.budget), flush=False)
    loss = run_test(x.args, **x.kwargs)
    optimizer.tell(x, loss)
    if n % 100 == 99:
        recommendation = optimizer.provide_recommendation()
        print(recommendation.loss, recommendation.args)
        write_nnue(**recommendation.kwargs)
        os.system("cp nn-tune.nxxx nn-recom-{}-{}-{}.nxxx".format(*recommendation.args))

recommendation = optimizer.provide_recommendation()
print(recommendation.value)
# Needed to store final network
run_test(recommendation.args, **recommendation.kwargs)
