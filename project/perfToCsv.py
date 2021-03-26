import csv
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

with open('perf_basic.csv', 'w', newline='') as csvfile:
    parameter_list = ['task-clock', 'cycles', 'instructions', 'L1-dcache-loads', 'L1-dcache-load-misses', 'LLC-loads', 'LLC-load-misses']
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'task-clock (mse)', 'CPUs utilized', 'Cycles', 'Instructions', 'L1-dcache-loads', 'L1-dcache-load-misses', 'LLC-loads', 'LLC-load-misses'])
    for filename in os.listdir(os.getcwd() + '/outputs/basic'):
        with open(os.getcwd() + '/outputs/basic/' + filename, 'r') as file:
            lines = file.readlines()
            data_list = []
            data_list.append(filename)
            for index, line in enumerate(lines):
                line_list = str.split(line)
                if len(line_list) != 0 and line_list[1] in parameter_list:
                    data_list.append(line_list[0].replace(',', ''))
                    if line_list[1] == 'task-clock':
                        data_list.append(line_list[4])

            writer.writerow(data_list)

with open('perf_smid.csv', 'w', newline='') as csvfile:
    parameter_list = ['fp_arith_inst_retired.scalar_single', 'fp_arith_inst_retired.scalar_double', 'fp_arith_inst_retired.128b_packed_single', 'fp_arith_inst_retired.256b_packed_single']
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'scalar_single', 'scalar_double', '128b_packed_single', '256b_packed_single'])
    for filename in os.listdir(os.getcwd() + '/outputs/simd'):
        with open(os.getcwd() + '/outputs/simd/' + filename, 'r') as file:
            lines = file.readlines()
            data_list = []
            data_list.append(filename)
            for index, line in enumerate(lines):
                line_list = str.split(line)
                if len(line_list) != 0 and line_list[1] in parameter_list:
                    data_list.append(line_list[0].replace(',', ''))

            writer.writerow(data_list)

with open('perf_time.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Average time per run (us)', 'Input resolution'])
    for filename in os.listdir(os.getcwd() + '/outputs/simd'):
        with open(os.getcwd() + '/outputs/time/' + filename, 'r') as file:
            lines = file.readlines()
            data_list = []
            data_list.append(filename)
            for index, line in enumerate(lines):
                if(index == 2):
                    line_list = str.split(line)
                    data_list.append(line_list[4])
                elif(index == 3):
                    line_list = str.split(line)
                    data_list.append(line_list[2])

            writer.writerow(data_list)