import argparse
parser = argparse.ArgumentParser(description='Signal only Fit')
parser.add_argument('--index', type=int, default=1, help='Index of the toy MC')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--config', type=str, default='test.yml', help='Config file')

parser.print_help()
args = parser.parse_args()
index=args.index
config_file=args.config
update = True
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import time
import sys
import tensorflow as tf
tf.get_logger().setLevel('INFO')
sys.path.append('/software/pc24403/tfpcbpggsz')  
from core import *


time1 = time.time()
config = ConfigLoader(config_file)
config.load_config()
config.load_each()
config.load_data('mc')
time2 = time.time()
config.load_norm()
time3 = time.time()
config.load_data()
time4 = time.time()
config.load_mass_pdfs()
config.update_yields()
time5 = time.time()

fitter = fit(config) 
fit_result, means, errors = fitter.fit()

##mg = m.scipy()
#
## Perform the scan
#if False:
#  contours = scanner(mg, 5, 45)
#  np.save('contours.npy', contours)

logpath = '/software/pc24403/tfpcbpggsz/test'
if os.path.exists(logpath) == False:
    os.mkdir(logpath)

with open(f'{logpath}/simfit_output_{index}.txt', 'w') as f:
    print(fit_result, file=f)
    print("Means", means['x0'], means['x1'], means['x2'], means['x3'], means['x4'], means['x5'], file=f)
    print("Errors", errors['x0'], errors['x1'], errors['x2'], errors['x3'], errors['x4'], errors['x5'], file=f)

time6 = time.time()

plot_result = plotter(fitter)
plot_result._path = '/software/pc24403/tfpcbpggsz/test/fig'
plot_result.plot_and_save()
time7 = time.time()

print(f'Load MC {time2-time1} seconds')
print(f'Load MC normalisation {time3-time2} seconds')
print(f'Load Data {time4-time3} seconds')
print(f'Load Mass PDFs {time5-time4} seconds')
print(f'Minimise and Save output  {time6-time5} seconds')
print(f'Plot and Save output  {time7-time6} seconds')
print(f'Total time: {time7-time1} seconds')