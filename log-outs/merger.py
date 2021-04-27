import os
from pathlib import Path
import re

parser = re.compile(r'^(?P<testcase>[\w_\d]*)(_BS(?P<BS>\(\d*, \d*\))){0,1} (?P<dims>\(\d* x \d*\)): (?P<time>\d*\.\d*)( us){0,1}$')

with open ('time.csv', 'w') as c:
    c.write("model, test_case, trials, image, blocksize_x, blocksize_y, rows, cols, total_pixels, avg_time(us)\n")
    for log_file in os.listdir('time_based'):
        with open('time_based/' + log_file) as f:
            x = f.readlines()
            trial_info = x[2].split(' ');
            res = parser.search(x[4])

            model, test_case = res.group('testcase').split('_')
            if model == 'GPGPU':
                b_x, b_y = res.group('BS')[1:-1].split(', ')
            else:
                b_x = ""
                b_y = ""
                
            ix, iy = res.group('dims')[1:-1].split(' x ')
            time = res.group('time')
            csv_str = f"{model}, {test_case}, {trial_info[1]}, {trial_info[4].split('(')[0][7:]}, {b_x}, {b_y}, {ix}, {iy}, {int(ix) * int(iy)}, {time}\n"
            c.write(csv_str)