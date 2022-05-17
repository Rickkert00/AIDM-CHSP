import time

import numpy as np
from minizinc import Instance, Model, Solver

def solve(linear_problems, solver_string,solver, i, do_print=False):
    if do_print:
      print(f"\nCurrently solving {str(i)}.dzn with solver {solver_string}")

    model = Model("../neil_paper_data_code/wallace-hoist_cpaior20-submission.mzn")
    params = linear_problems[i]
    for param, v in params.items():
      model[param] = v

    instance = Instance(solver, model)
    args = {'output-time': True, 'solver-statistics': True, 'time-limit': 300000}
    start_time = time.time()
    results = instance.solve(intermediate_solutions=False, **args)
    duration = time.time() - start_time
    if do_print:
        print(results.statistics)
        print(results.solution)
        print(f'Duration: {duration:.3f}s')
        if duration >= args['time-limit'] / 1000:
          print(f"Example {i} timed out!")
    if results.solution is None:
      return None, None
    return results, params

def main():
    NUM_EXAMPLES = 5000
    solver_string = 'gecode'

    solver = Solver.lookup(solver_string)
    solutions = []

    param_file = "instances/linearproblems.npy"
    linear_problems = np.load(param_file, allow_pickle=True)
    for i in range(NUM_EXAMPLES + 1):
      results, params = solve(linear_problems, solver_string, solver,  i)
      if results is not None:
        output_dict = {k: getattr(results.solution, k) for k in results.solution.__dict__ if k[0] != '_'}
        output_dict['jobs'] = int(str(results).split('jobs ')[1])
        solutions.append((params, output_dict))
        if i %10 == 1:
          print("Temp saving. Currently at index", i, 'of total', NUM_EXAMPLES+1)
              # temp save
          np.save('instances/linear_solutions.npy', solutions)

    np.save('instances/linear_solutions.npy', solutions)


if __name__ == '__main__':
    main()