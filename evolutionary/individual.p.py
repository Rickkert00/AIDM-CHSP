import json
import math

import numpy as np

from evolutionary.problem_instance import ProblemInstance


class Individual:
    def __init__(self, genotype: np.ndarray, instance: ProblemInstance):
        """:param genotype: the order of moves between between successive tanks 0: T0 to T1, 1: T1 to T2, etc."""
        if genotype[0] != 0:
            raise Exception("First element of genotype should always be zero")
        self.genotype = genotype
        self.removal_times = None
        self.fitness = math.inf
        self.instance = instance
        self.is_valid = False

    @staticmethod
    def ind_from_removal_times(removal_times: np.ndarray, instance: ProblemInstance):
        moves_order = np.argsort(removal_times)
        return Individual(moves_order, instance)

    def compute_fitness(self):
        nop = self.genotype.size
        z = np.zeros((2 * nop - 1, 2 * nop - 1), dtype=np.int64)
        print(z.shape)
        # TODO fix indexing cannot reach to 2 * nop - 2
        for k in range(1, 2 * nop - 1):
            for i in range(k):
                if k % nop != 0:
                    beta = np.max(np.argwhere(self.genotype == self.genotype[k % nop] - 1))
                else:
                    beta = -1
                print(beta)
                # beta should be bigger than i and smaller than k otherwise not valid
                if beta <= i or beta >= k:
                    beta = None

                # TODO could be that d should be indexed with k instead of k - 1
                temp_k_min1 = self.genotype[(k - 1) % nop]
                temp_k_min1_to = (temp_k_min1 + 1) % nop
                temp_k = self.genotype[k % nop]
                empty_move = z[i, k - 1] + self.instance.d[temp_k_min1] + self.instance.c[temp_k_min1_to, temp_k]
                # We do not take into account previous soaking
                if beta is None:
                    z[i, k] = empty_move
                else:
                    i_temp = self.genotype[i % nop]
                    i_temp_to = i_temp + 1
                    full_move = z[i, beta] + self.instance.d[i_temp] + self.instance.a[i_temp_to]
                    z[i, k] = max(full_move, empty_move)

        self.removal_times = np.zeros(nop)
        for k in range(nop):
            self.removal_times[k] = z[0, k]

        # self.fitness = z
        # TODO compute fitness

    def __str__(self):
        return json.dumps({
            "Genotype:": str(self.genotype),
            "Removal times:": str(self.removal_times),
            "Fitness:": self.fitness
        }, indent=4)


if __name__ == '__main__':
    # This example instance is 1.dzn
    example_instance = ProblemInstance(
        np.array([159, 343, 50]),
        np.array([214, 401, 72]),
        np.array([11, 13, 13, 11]),
        np.array([6, 0, 6, 12,
                  12, 6, 0, 6,
                  18, 12, 6, 0,
                  0, 6, 12, 18])
    )
    ind = Individual.ind_from_removal_times(np.array([0, 170, 183, 246]), example_instance)
    print(ind)
    ind.compute_fitness()
    print(ind)
