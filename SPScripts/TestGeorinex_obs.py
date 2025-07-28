import georinex as gr
import os


file_path = '../data/TutData/aspa3240.24o'

ds = gr.load(file_path)
print(ds)
print(ds.coords['time'].dtype)
obs_df = ds.to_dataframe()
print(obs_df.index.get_level_values('time').dtype)
