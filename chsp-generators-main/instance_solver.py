import time

from minizinc import Instance, Model, Solver, Result


NUM_EXAMPLES = 5000
solver_string = 'gecode'

solver = Solver.lookup(solver_string)
for i in range(1, NUM_EXAMPLES + 1):
    print(f"\nCurrently solving {str(i)}.dzn with solver {solver_string}")

    model = Model("../neil_paper_data_code/wallace-hoist_cpaior20-submission.mzn")
    model.add_file(f"instances/linear/{str(i)}.dzn")

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

    with open(f"instances/linear_solutions/{str(i)}.txt", "w") as fw:
        fw.write(str(results.solution) +
                 f'\nDuration: {duration:.3f} seconds' +
                 f'\nSolver: {solver_string}')

    # print(results.solution)
    # row = [i] + [(int(times.total_seconds() * 1000), amount) for times, amount in zip(results.statistics['time'], results.statistics['weekends'])]

    # print(results)

