import gc
from physics import seq2seq as pjt
from fun import fancy_print
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pycwt as wavelet
import math
import logging
import matlab.engine
import matlab
from tqdm import tqdm
from pathlib import Path

"""
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
"""

# Manually set matplotlib backend to fix memory leaks
# Images must be saved in .png format as well
matplotlib.use('Agg')

# Get data
fancy_print('Getting data', fg='chartreuse')
#dpk = pjt.extract_data(plant='Afl', run_id=5, search='files', process=False)
tp = r'/Users/dannerd/seq2seq/data/Afl_BT120_run5_2023-01-15--041341-UTC1.datapkg'
dpk = pjt.load_datapackage(tp)
dpk.onboard()
df = dpk.source.data
df = df.sort_index()

# Start MATLAB engine
fancy_print('Starting MATLAB engine', fg='chartreuse')
eng = matlab.engine.start_matlab()

# Describe data
Columns = df.columns
print(f"{Columns}")
print(df.describe)

# Get data shape and sample freq (fs_hz)
fancy_print('Getting data shape', fg='chartreuse')
fs_hz = dpk.temporal.fs_hz
dt = 1 / fs_hz
N = df.shape[0]
COLS = df.shape[1]
T = np.arange(0, N) * dt
fancy_print(f"dt = {dt}, Num Cols = {COLS}", fg='chartreuse')

# Setup wavelets
fancy_print('Setting up wavelets', fg='chartreuse')
mother = wavelet.Morlet(6)
s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 secs = .5 seconds
dj = 1 / 12  # Twelve sub-octaves per octave
J = 7 / dj  # Seven powers of two with dj sub-octaves
ARRAY_SIZE = 4096
BATCHES = math.floor(N / ARRAY_SIZE)

# Setup structures for stats and initialize counter "l", value "t"
std = np.zeros([BATCHES, COLS])
var = np.zeros([BATCHES, COLS])
alpha = np.zeros([BATCHES, COLS])
df_notrend = pd.DataFrame(index=T[0:ARRAY_SIZE], columns=Columns)
df_norm = pd.DataFrame(index=T[0:N], columns=Columns)

# Setup image file paths
p = Path(__file__)
p = p.parents[0]
ts_path = p.joinpath('dqs','timeseriespicstmp3')
coherence_path = p.joinpath('dqs','coherencepicstmp3')
if not ts_path.exists():
    Path.mkdir(ts_path, parents=True)
if not coherence_path.exists():
    Path.mkdir(coherence_path, parents=True)

logging.basicConfig(
    level=logging.DEBUG,  # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename= p.joinpath('wcoherence4_cc4.log'),  # Specify the filename for the log file
    filemode="w",  # "w" to overwrite the file on each run, "a" to append
)
logging.debug("Entering primary loop structure for coherence calculations.")
# Outer loop
fancy_print('Entering loops', fg='chartreuse')
for col_name1, col_ldx in tqdm(zip(Columns[:], range(COLS))):
    for batch_idx in tqdm(range(BATCHES)):
   
        # Detrend and normalize
        t = T[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1)]
        p = np.polyfit(
            t,
            df.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_ldx].values.astype(float),
            0,
        )
        df_notrend[col_name1] = \
            df.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_ldx].values.astype(float) - np.polyval(p, t)
    
        # Test 0
        try:
            if df_notrend[col_name1].isna().sum():
                raise ValueError
        except ValueError:
            logging.critical(
                f"Trend removal produced NaNs for Column {col_name1}, batch {batch_idx}")
    
        # Handle 0-std case
        std[batch_idx, col_ldx] = df_notrend.iloc[:, col_ldx].std()  # Standard deviation
        var[batch_idx, col_ldx] = std[batch_idx, col_ldx] ** 2  # Variance
        if np.isclose(std[batch_idx, col_ldx], 0):
            logging.critical(f"Std[{batch_idx},{col_ldx}] = 0, setting to 1.0")
            std[batch_idx, col_ldx] = 1.0
            var[batch_idx, col_ldx] = 1.0
    
        # Test 1
        try:
            # Normalized dataset
            df_norm.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_ldx] = df_notrend[col_name1] / std[batch_idx, col_ldx]
            for m in range(0, ARRAY_SIZE):
                if math.isinf(df_norm.iloc[batch_idx * ARRAY_SIZE + m, col_ldx]):
                    raise ValueError(f"{df_norm[batch_idx * ARRAY_SIZE + m, col_ldx]}: infinite value detected")
        except OverflowError(batch_idx, Columns[col_name1]):
            logging.error(
                f"Exception Overflow Error in batch {batch_idx} and Column: {col_ldx}: {Columns[col_name1]}, batch {batch_idx}")
        except ValueError as e:
            logging.error(e)
    
        # Test 2
        try:
            if df_norm.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_ldx].isna().sum():
                raise ValueError
        except ValueError:
            logging.error(
                f"Normalization produced NaNs for Column {col_name1}, batch {batch_idx}")
    
        # Test 3
        try:
            # Lag-1 autocorrelation for red noise
            alpha[batch_idx, col_ldx], _, _ = wavelet.ar1(df_norm.iloc[t, col_ldx])
        except BaseException as e:
            logging.warning(
                f"ARL estimation is unstable for Column {col_name1} and exception {e}, batch {batch_idx}")
        alpha[batch_idx, col_ldx] = 1



    for col_name2, col_idx in tqdm(zip(Columns[col_ldx:], range(col_ldx, COLS))):
   
        for batch_idx in tqdm(range(BATCHES)):
    
            fancy_print(f'\tl: {col_ldx}, i: {col_idx}, k: {batch_idx}', fg='medium_grey')
    
            # Detrend and normalize
            t = T[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1)]
            #repeat scaling of time series from second columns    
            p = np.polyfit(
                t,
                df.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_idx].values.astype(float),
                0,
            )
            df_notrend.iloc[:, col_idx] = \
                df.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_idx].values.astype(float) - np.polyval(p, t)
    
            # Test 4
            try:
                if df_notrend.iloc[:, col_idx].isna().sum():
                    raise ValueError
            except ValueError:
                logging.error(
                    f"Trend removal produced NaNs for Column {Columns[col_idx]}, batch {batch_idx}")
    
            # Handle 0-std case
            std[batch_idx, col_idx] = df_notrend.iloc[:, col_idx].std()  # Standard deviation
            var[batch_idx, col_idx] = std[batch_idx, col_idx] ** 2  # Variance
            if np.isclose(std[batch_idx, col_idx], 0):
                logging.critical(f"Std[{batch_idx},{col_idx}] = 0, setting to 1.0")
                std[batch_idx, col_idx] = 1.0
                var[batch_idx, col_idx] = 1.0
    
            # Test 5
            try:
                # Normalized dataset
                df_norm.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_idx] = df_notrend.iloc[:, col_idx] / std[batch_idx, col_idx]
                for m in range(0, ARRAY_SIZE):
                    if math.isinf(df_norm.iloc[batch_idx * ARRAY_SIZE + m, col_idx]):
                        raise ValueError(f"{df_norm.iloc[batch_idx * ARRAY_SIZE + m, col_idx]}: infinite value detected")
            except OverflowError:
                logging.error(
                    f"Exception Overflow Error in row {batch_idx * ARRAY_SIZE + m} and Column {col_idx}: {Columns[col_name2]}")
            except ValueError as e:
                logging.error(e)
    
            # Test 6
            try:
                if df_norm.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_idx].isna().sum():
                    raise ValueError
            except ValueError:
                logging.error(
                    f"Normalization produced NaNs for Column {col_name2}, batch {batch_idx}")
            except BaseException as e:
                logging.error(
                    f"Normalization step for Column {col_name2}, batch {batch_idx}")
    
            # Test 7
            try:
                # Lag-1 autocorrelation for red noise
                alpha[batch_idx, col_idx], _, _ = wavelet.ar1(
                    df_norm.iloc[batch_idx * ARRAY_SIZE:(batch_idx + 1) * ARRAY_SIZE, col_idx])
            except BaseException as e:
                logging.warning(
                    f"Arl-1 produced unstable estimate for Column {col_name2}, batch {batch_idx}")
                alpha[batch_idx, col_idx] = 1
    
            # Write values to MATLAB workspace
            eng.workspace['dfcoll'] = matlab.double(
                df_norm.iloc[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1), col_ldx].values.astype(float)
            )
            #pd.DataFrame(df_norm.iloc[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1), col_ldx].values.astype(float)).to_csv(f"col_ldx{col_ldx}.csv", index=False)

            eng.workspace['dfcoli'] = matlab.double(
                df_norm.iloc[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1), col_idx].values.astype(float)
            )
            #pd.DataFrame(df_norm.iloc[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1), col_idx].values.astype(float)).to_csv(f"col_idx{col_idx}.csv", index=False)

            eng.workspace['t'] = matlab.double(
                np.array(T[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1)])
            )
            #pd.DataFrame(np.array(T[batch_idx * ARRAY_SIZE:ARRAY_SIZE * (batch_idx + 1)])).to_csv(f"batch_idx{batch_idx}.csv", index=False)
            
            filename = ts_path.joinpath( f"timeseries_pics[{col_ldx}][{col_idx}]batch[{batch_idx}].png")

            # Plot via MATLAB
            dfcollcopy = eng.workspace['dfcoll']
            dfcolicopy = eng.workspace['dfcoli']
            tcopy = eng.workspace['t']
            eng.eval("figure(1)")
            eng.workspace['filename'] = str(filename)
            eng.eval("plot(t, dfcoll, 'k', t,  dfcoli, 'r')")
            pystr = f"Col: {Columns[col_ldx]}, + Col: {Columns[col_idx]} Batch: [{batch_idx}]"
            eng.workspace['str'] = pystr
            eng.eval("title(str)")
    
            # Plot via Matplotlib
            plt.plot(np.squeeze(tcopy), np.squeeze(dfcollcopy), np.squeeze(tcopy), np.squeeze(dfcolicopy))
            plt.xlabel('Time (sec)')
            plt.title(pystr)
            plt.savefig(filename, format="png")
    
            # Wavelet coherence
            SamplePeriod = eng.seconds(1 / fs_hz)
            result = eng.wcoherence(
                dfcollcopy,
                dfcolicopy,
                SamplePeriod,
#                'numscales',
#                matlab.double(16),
                nargout=4,
            )
            wavecoherence, crossspectrum, period, coi = result
            wavecoherencepy = np.array(wavecoherence)
            min_indices = np.where(wavecoherencepy < 0)
            wavecoherencepy[min_indices] = 0
            max_indices = np.where(wavecoherencepy > 1)
            wavecoherencepy[max_indices] = 1
            crossspectrumpy = np.array(crossspectrum)
            wavephase = np.angle(crossspectrumpy, deg=True)
    
            # Plot via MATLAB
            #eng.eval("figure(2)")
            # eng.close_all
            #eng.eval(f"wcoherence(dfcoll,dfcoli,{1 / fs_hz})")
    
            # Contour plot via Matplotlib
            x = np.float64(tcopy)
            y = range(0, wavecoherencepy.shape[0])
            mscales = matlab.int32(y)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, wavecoherencepy, levels=20, cmap=plt.cm.RdGy)
            plt.colorbar()
            logging.debug(f"Minimum coherence value: {np.min(wavecoherencepy)} for batch: {batch_idx}, cols: {col_ldx},{col_idx}")
            logging.debug(f"Maximum coherence value: {np.max(wavecoherencepy)} for batch: {batch_idx}, cols: {col_ldx},{col_idx}")
    
            # Adding labels and title
            plt.ylabel('Scales')
            plt.xlabel('Time (sec)')
            plt.title(
                f"Contour Plot of Wavelet Coherence for {Columns[col_ldx]}, {Columns[col_idx]} at batch {batch_idx} of {BATCHES}")
            filename = coherence_path.joinpath(f"coherence_pics[{col_ldx}][{col_idx}]batch[{batch_idx}].png")
            plt.savefig(filename, format="png")
    
            # Here to prevent additional memory leaks
            plt.show()
            plt.clf()
            plt.close('all')
            gc.collect()
        
            fancy_print(f"Completed batch {batch_idx}, Col {col_ldx}, Col {col_idx}",fg='chartreuse')
        fancy_print(f"Completed all batches for Col {col_ldx}, Col {col_idx}", fg='chartreuse')
    fancy_print(f"Completed all Column Comparisons for Col {col_ldx}, Cols 0-66", fg='chartreuse')
eng.quit()
