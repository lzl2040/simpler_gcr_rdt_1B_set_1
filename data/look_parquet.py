import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np

file_path = "/Data/lerobot_data/simulated/libero_spatial_no_noops_lerobot/merged.parquet"
table = pq.read_table(
    file_path,
    use_threads=True,  # 启用多线程
    memory_map=True    # 内存映射文件加速
)
list_parquet = table.to_pylist() # all frames len
pandas_data = table.to_pandas()
action_numpy = np.stack(pandas_data["action"].values)
print(action_numpy.shape)