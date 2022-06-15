import os
import time
from multiprocessing import Pool

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
def run(i):
    results, params = solve(linear_problems, solver_string, solver, i)
    if results is not None:
        output_dict = {k: getattr(results.solution, k) for k in results.solution.__dict__ if k[0] != '_'}
        output_dict['jobs'] = int(str(results).split('jobs ')[1])
        return (params, output_dict)
    else:
        return None,None
def main():
    global linear_problems, solver_string, solver
    solver_string = 'gecode'; num=4
    param_file = f"instances/linearproblems_{num}.npy"
    solution_file = f'instances/linear_solutions_{num}.npy'
    solver = Solver.lookup(solver_string)
    solutions = []
    if os.path.exists(solution_file):
        solutions = np.load(solution_file, allow_pickle=True).tolist()
    linear_problems = list(np.load(param_file, allow_pickle=True).item().values())
    solution_file = f'instances/linear_solutions_{3}.npy'
    print(solutions[-1:],np.load(solution_file, allow_pickle=True).tolist()[-1:])
    print(solutions[0],np.load(solution_file, allow_pickle=True).tolist()[0])
    assert solutions[-1:] == np.load(solution_file, allow_pickle=True).tolist()[-1:]
    parallel = 10
    print(f"Running seed {num} Start at", len(linear_problems))
    for i in range(len(solutions), len(linear_problems), parallel):
        with Pool(parallel) as p:
            ret = p.map(run, range(i, i + parallel))
            solutions.extend(ret)
        print("Temp saving. Currently at index", i, 'of total', len(linear_problems))
        np.save(solution_file, solutions)

    np.save(solution_file, solutions)


if __name__ == '__main__':
    main()