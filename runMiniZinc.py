#!/usr/bin/python3

# number of cpus
# SBATCH --cpus-per-task=1
# ammount of RAM memory
# SBATCH --mem=1024
# mail at the end of the task
# SBATCH --mail-type=END
# SBATCH --time=1:00:00
import asyncio
import os

import numpy as np
from minizinc import Instance, Model, Solver
import time, sys

# Cluster imports

# from searchSpace import instanceGeneration
# from searchSpace import runSearchSpace

def main():
  pass
  # Cluster imports
  # import runSearchSpace

  # runSearchSpace.run_search_space_experiments(int(sys.argv[1]))
  # runSearchSpace.run_search_space_experiments(sys.argv[1])

  # print("Command line argument: ", int(sys.argv[1]))



def example_use():
  # Generate data and run a single instance
  data_filename_to_use = "generated_instance_1"
  # data_filename_to_use = "1-1-1"
  directory = "./dataFiles/"
  datafile_location = directory + data_filename_to_use + ".dzn"
  if not os.path.exists(directory):
    os.makedirs(directory)
  print("Using file", datafile_location)
  # problem_instance = instanceGeneration.generate_data(12, 9, 1, 1, 1, 1, 1, 1)
  # problem_instance.write_to_datafile(datafile_location)
  run_instance(datafile_location, "", "gecode")

  # Generate datafile for the multipliers used in the experiments
  # additional_data_file_location = "./dataFiles/moreModelData.dzn"
  # instanceGeneration.generate_additional_data(additional_data_file_location, 1, 1, 1)


def run_experiment(data_filename_basis, solver_name, variable_parameters, variable_parameter_values,
                   number_of_instances_per_parameter_value, parameters):
  # This method does not use the same instances but altered for each variable parameter setting
  # Instead it uses the same underlying distributions

  # Generate necessary datafile location names
  datafile_locations = []
  number_of_instances = number_of_instances_per_parameter_value * len(variable_parameter_values)
  for i in range(number_of_instances):
    datafile_locations.append(get_datafile_location(data_filename_basis + str(i + 1)))

  # Generate necessary datafiles
  for i in range(len(variable_parameter_values)):

    # Set variable parameters
    for var in range(len(variable_parameters)):
      parameters[variable_parameters[var]] = variable_parameter_values[i][var]
      print("Parameter ", variable_parameters[var], " set to ", variable_parameter_values[i][var])

    # Generate problem instances and write them to datafiles
    for j in range(number_of_instances_per_parameter_value):
      problem_instance = instanceGeneration.generate_data(parameters[0], parameters[1], parameters[2], parameters[3],
                                                          parameters[4], parameters[5], parameters[6], parameters[7])
      problem_instance.write_to_datafile(datafile_locations[i * number_of_instances_per_parameter_value + j])

  # Create place to store experimental results
  objective_values = [[0.0 for i in range(number_of_instances_per_parameter_value)] for j in range(len(variable_parameter_values))]
  solve_times = [[0.0 for i in range(number_of_instances_per_parameter_value)] for j in range(len(variable_parameter_values))]

  # Run experiments
  for i in range(len(variable_parameter_values)):
    for j in range(number_of_instances_per_parameter_value):
      print("-------Instance array index: ", i * number_of_instances_per_parameter_value + j)
      objective, solve_time = run_instance(datafile_locations[i * number_of_instances_per_parameter_value + j], "", solver_name)
      print("Optimal objective value: ", objective)
      print("Time to solve: ", solve_time)

      objective_values[i][j] = objective
      solve_times[i][j] = solve_time

  return objective_values, solve_times

def run_instance(data_file, additional_data_file, solver_name, do_print=True):
  return asyncio.run(run_instance_async(data_file, additional_data_file, solver_name, do_print=True))

async def run_instance_async(data_file, solver_name, additional_data_file=None, do_print=True):
  # Get the model/data files
  model = Model("modelFiles/wallace-hoist_cpaior20-submission.mzn")
  model.add_file(data_file)
  if additional_data_file is not None: model.add_file(additional_data_file)

  # Get the solver
  solver = Solver.lookup(solver_name)
  print("Solver used:", solver.name)
  print("path", os.getcwdb())
  # Create instance and solve
  instance = Instance(solver, model)

  result = await instance.solve_async(intermediate_solutions=True)
  # Print all results
  if do_print:
    for i in range(len(result)):
      print("-------Result:\n", result[i])
    # pass

  # print("Time it took to solve: ", (endTime - startTime))
  # print("Best value: ", result[len(result)-1])
  if len(result) == 0:
    print("Warning no results found")
    return -1
  return result[len(result) - 1].objective, result.statistics["solveTime"]


def get_datafile_location(filename):
  return "/data/dzn/" + filename + ".dzn"


if __name__ == "__main__":
  DATA_FOLDER = "data/dzn/"
  files = os.listdir(DATA_FOLDER)
  objs = []
  secs = []
  for f in files[:1]:
    obj, sec = run_instance(DATA_FOLDER+f, "", "gecode")
    print('obj', obj,'sec', sec)
    objs.append(obj)
    secs.append(secs)
  # np.save('data/solved/dzn_output.npy', {'obj':objs, 'sec':secs})
  example_use()