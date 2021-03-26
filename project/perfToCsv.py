import csv
import os

with open('perf_stat.csv', 'w', newline='') as csvfile:
    parameter_list = ['task-clock', 'cycles', 'instructions', 'L1-dcache-loads', 'L1-dcache-load-misses', 'LLC-loads', 'LLC-load-misses',
        'fp_arith_inst_retired.scalar_single', 'fp_arith_inst_retired.scalar_double', 'fp_arith_inst_retired.128b_packed_single', 'fp_arith_inst_retired.256b_packed_single'
    ]
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'task-clock (mse)', 'CPUs utilized', 'Cycles', 'Instructions', 'IPC', 'L1-dcache-loads', 'L1-dcache-load-misses', 'L1-dcache miss rate', 'LLC-loads', 'LLC-load-misses', 'LLC miss rate',
        'scalar_single', 'scalar_double', '128b_packed_single', '256b_packed_single',
        'Average time per run (us)', 'Input resolution'
    ])
    for filename in os.listdir(os.getcwd() + '/outputs/basic'):
        with open(os.getcwd() + '/outputs/basic/' + filename, 'r') as basic_file, open(os.getcwd() + '/outputs/simd/' + filename, 'r') as simd_file, open(os.getcwd() + '/outputs/time/' + filename, 'r') as time_file:
            basic_row = basic_file.readlines()
            simd_row = simd_file.readlines()
            time_row = time_file.readlines()
            data_list = []
            data_list.append(filename)
            for line in basic_row:
                line_list = str.split(line)
                if len(line_list) != 0 and line_list[1] in parameter_list:
                    data_list.append(line_list[0].replace(',', ''))
                    if line_list[1] == 'task-clock':
                        data_list.append(line_list[4])
                    elif line_list[1] == 'instructions':
                        data_list.append(line_list[3])
                    elif line_list[1] == 'L1-dcache-load-misses':
                        data_list.append(line_list[3])
                    elif line_list[1] == 'LLC-load-misses':
                        data_list.append(line_list[3])

            for line in simd_row:
                line_list = str.split(line)
                if len(line_list) != 0 and line_list[1] in parameter_list:
                    data_list.append(line_list[0].replace(',', ''))

            for index, line in enumerate(time_row):
                if(index == 2):
                    line_list = str.split(line)
                    data_list.append(line_list[4])
                elif(index == 3):
                    line_list = str.split(line)
                    data_list.append(line_list[2])

            writer.writerow(data_list)
