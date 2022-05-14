import time

import numpy as np
from minizinc import Instance, Model, Solver, Result

def get_params_of_file(file):
  params = {}
  with open(file) as f:
    for line in f:
      split = line.split()
      param, val = split[0], split[2][:-1]
      params[param] = int(val)
  return params

def main():
    NUM_EXAMPLES = 5000
    solver_string = 'gecode'

    solver = Solver.lookup(solver_string)
    for i in range(1, NUM_EXAMPLES + 1):
        print(f"\nCurrently solving {str(i)}.dzn with solver {solver_string}")

        model = Model("../neil_paper_data_code/wallace-hoist_cpaior20-submission.mzn")
        param_file = f"instances/linear/{str(i)}.dzn"
        model.add_file(param_file)
        params = get_params_of_file(param_file)
        input_dict = {}

        instance = Instance(solver, model)
        args = {'output-time': True, 'solver-statistics': True, 'time-limit': 300000}
        start_time = time.time()
        results = instance.solve(intermediate_solutions=False, **args)
        duration = time.time() - start_time
        print(results.statistics)
        print(results.solution)
        print(f'Duration: {duration:.3f}s')
        if duration >= args['time-limit'] / 1000:
            print(f"Example {i} timed out!")
        if results.solution is None:
            continue

        output_dict = {k: getattr(results.solution, k) for k in results.solution.__dict__ if k[0] != '_'}
        output_dict['jobs'] = int(str(results).split('jobs ')[1])
        np.save(f'instances/linear_solutions/{str(i)}.npy', [input_dict, output_dict])
        # with open(f"instances/linear_solutions/{str(i)}.txt", "w") as fw:
        #     fw.write(str(results.solution) +
        #              f'\nDuration: {duration:.3f} seconds' +
        #              f'\nSolver: {solver_string}')

        # print(results.solution)
        # row = [i] + [(int(times.total_seconds() * 1000), amount) for times, amount in zip(results.statistics['time'], results.statistics['weekends'])]

        # print(results)
if __name__ == '__main__':
    main()