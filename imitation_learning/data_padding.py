from torchsummary import summary

from imitation_learning.neural_network import BoundPredictor


class ProblemInstance:


    def __init__(self, num_tanks, max_num_jobs_in_parallel, tmin, tmax, e, f, hoists, multiplier, capacity):
        self.num_tanks = num_tanks
        self.max_num_jobs_in_parallel = max_num_jobs_in_parallel
        self.tmin = tmin
        self.tmax = tmax
        self.e = e
        self.f = f
        self.hoists = hoists
        self.multiplier = multiplier
        self.capacity = capacity

    def __str__(self):
        return f"Number of tanks: {self.num_tanks}\n" \
               f"Max number of jobs in parallel {self.max_num_jobs_in_parallel}\n" \
               f"tmin: {self.tmin}\n" \
               f"tmax: {self.tmax}\n" \
               f"e: {self.e}\n" \
               f"f: {self.f}\n" \
               f"hoists: {self.hoists}\n" \
               f"multiplier: {self.multiplier}\n" \
               f"capacity: {self.capacity}"



def get_data_instance(pad_size, INF):
    num_tanks = 12  # Ninner
    max_num_jobs_in_parallel = 9  # J
    tmin = [2400, 1800, 600, 600, 600, 600, 1200, 600, 1200, 600, 900, 2400]
    tmax = [INF, 2400, 900, 2400, 900, 1200, 1800, 900, 1450, 900, 1200, 4200]
    e = [[0, 0, 143, 133, 125, 111, 100, 90, 80, 64, 52, 41, 30],
         [0, 143, 0, 10, 18, 32, 43, 53, 64, 79, 91, 102, 12],
         [0, 133, 10, 0, 8, 22, 33, 43, 53, 69, 81, 92, 11],
         [0, 125, 18, 8, 0, 14, 25, 35, 45, 61, 73, 84, 10],
         [0, 111, 32, 22, 14, 0, 11, 21, 31, 47, 59, 70, 88],
         [0, 100, 43, 33, 25, 11, 0, 10, 20, 36, 48, 59, 77],
         [0, 90, 53, 43, 35, 21, 10, 0, 10, 26, 38, 49, 67],
         [0, 80, 64, 53, 45, 31, 20, 11, 0, 15, 28, 39, 57],
         [0, 64, 79, 69, 61, 47, 36, 26, 15, 0, 12, 23, 41],
         [0, 52, 91, 81, 73, 59, 48, 38, 28, 12, 0, 11, 29],
         [0, 41, 102, 92, 84, 70, 59, 49, 39, 23, 11, 0, 18],
         [0, 30, 120, 110, 102, 88, 77, 67, 57, 41, 29, 18, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    f = [0, 259, 186, 124, 130, 187, 126, 186, 131, 188, 127, 206, 146]
    hoists = 1
    multiplier = 1
    capacity = 1
    # we need to pad until pad_size
    tmin = tmin + [-1] * (pad_size - num_tanks)
    tmax = tmax + [-1] * (pad_size - num_tanks)
    e = [row + [-1]*(pad_size - num_tanks) for row in e]
    f = f + [-1] * (pad_size - num_tanks)
    return ProblemInstance(num_tanks, max_num_jobs_in_parallel, tmin, tmax, e, f, hoists, multiplier, capacity)



if __name__ == '__main__':
    pad_size = 60
    INF = 9999
    instance = get_data_instance(pad_size, INF)
    print(instance)
    predictor = BoundPredictor()
    summary(predictor, (200, 3785))