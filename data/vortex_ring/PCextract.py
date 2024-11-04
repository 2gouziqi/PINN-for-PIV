import struct
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def read_ply(file_path):
    try:
        with open(file_path, 'rb') as f:
            is_binary = False
            lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                lines.append(line)
                if line == 'end_header':
                    break
                if 'format binary' in line:
                    is_binary = True
            
            headers = []
            for line in lines:
                if line.startswith("property"):
                    headers.append(line.split()[-1])

            data = []
            if is_binary:
                while True:
                    byte_data = f.read(4 * len(headers))
                    if not byte_data:
                        break
                    data.append(struct.unpack('f' * len(headers), byte_data))
            else:
                start_idx = len(lines)
                data = pd.read_csv(file_path, delim_whitespace=True, skiprows=start_idx, header=None, names=headers, dtype=float)

            df = pd.DataFrame(data, columns=headers)
            return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":

    ply_file_path = "t1/point_cloud.ply"
    ply_df = read_ply(ply_file_path)

    if ply_df is not None:
        csv_file_path = os.path.splitext(ply_file_path)[0] + '.csv'
        ply_df.to_csv(csv_file_path, index=False)
        print(f"CSV file saved at: {csv_file_path}")
        
    else:
        print("Failed to read the PLY file.")
