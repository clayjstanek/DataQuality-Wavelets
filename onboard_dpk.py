# Import Project API
from physics import seq2seq as pjt

# Path to where datapackage is stored.
# path_to_dpk = r'C:\Users\cgu4\seq2seq\data\Spartanburg_2487_line18_run12952_2023-11-06--073024-UTC.datapkg'
# path_to_dpk = r'C:\Users\cgu4\seq2seq\data\Spartanburg_2487_line18_run13013_2023-11-09--203013-UTC.datapkg'
path_to_dpk = r'C:\Users\cgu4\seq2seq\data\Afl_BT120_lineBL22_run15_2023-04-27--203348-UTC.datapkg'

# Load the data package.
dpk = pjt.load_datapackage(path_to_dpk)

# Onboard the data package to bring source data into memory
dpk.onboard()

# Access raw data (contains entire data set and all the signals)
raw_data = dpk.source.data
print(raw_data.shape)

# Access statistical measurements for raw data
stats_raw = dpk.signal_info.stats
print(stats_raw.mean)
print(stats_raw.std)
print(stats_raw.quantiles)

# See process engineer recommended signals
ctrl_inputs = dpk.recommendations_expert.signal_selections.pre_pipeline.ctrl_inputs
unctrl_inputs = dpk.recommendations_expert.signal_selections.pre_pipeline.unctrl_inputs
outputs = dpk.recommendations_expert.signal_selections.pre_pipeline.outputs
print("Suggested controllable inputs are ", ctrl_inputs)
print("Suggested uncontrollable inputs are ", unctrl_inputs)
print("Suggested outputs are ", outputs)

# Access transformed data (contains cleansed data set after removing outliers (if any) and selected signals)
transformed_data = dpk.transformed.data
print(transformed_data.shape)

# Access statistical measurements for transformed data
stats_transformed = dpk.transformed.stats
print(stats_transformed.mean)
print(stats_transformed.std)
print(stats_transformed.quantiles)

# Access pipelines used to cleanse the data
pipeline_ops = dpk.recommendations_expert.pipeline.operations
print(pipeline_ops)
