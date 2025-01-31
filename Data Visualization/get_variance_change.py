import matplotlib.pyplot as plt
import numpy as np

def read_and_process_file(file_path):
    result_list = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            # Extract every second line
            for i in range(1, len(lines), 2):
                line = lines[i].strip()

                # Split the line into a list of integers
                integer_series = list(map(int, line.split()))

                result_list.append(integer_series)

        return result_list

    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    
def normalize_read_list(lst):
    for i, line in enumerate(lst):
        line_sum = sum(line)
        if line_sum != 0:  # Avoid division by zero
            lst[i] = [element / line_sum for element in line]

def variance_from_mean(arr):
    # Calculate the variance for each row
    return np.var(arr, axis=1).tolist()

file_name = "recorder.txt"
read_list = read_and_process_file(file_name)

if read_list is not None:
    normalize_read_list(read_list)
    np_read_list = np.array(read_list)

    var_res = variance_from_mean(np_read_list)
    x = [i for i in range(1, len(var_res) + 1)]

    print(var_res)

    fig, ax = plt.subplots()
    ax.plot(x, var_res)
    ax.set_xlabel("Combination Length (s)", fontname='Times New Roman', fontsize=12)
    ax.set_ylabel("Total Variance of $\overrightarrow{\mathrm{NR}}$ in Selected Combination Length", fontname='Times New Roman', fontsize=12)
    # ax.set_title("Variance Analysis", fontname='Times New Roman', fontsize=12)
    
    # Set font for tick labels
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Set font for the scientific notation offset text
    ax.yaxis.get_offset_text().set_fontname('Times New Roman')

    ax.grid()
    plt.savefig("variance_plot.png", dpi=300)
    plt.show()