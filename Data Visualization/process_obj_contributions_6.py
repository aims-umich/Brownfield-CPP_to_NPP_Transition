import pandas as pd

def process_obj_contributions(file_path="obj_contributions.txt", output_csv="sum_norm_obj_cont.csv"):
    # Open the file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    block_size = 16057  # Number of lines in each block
    columns = 22  # Number of integers per line
    current_line = 1  # Start after the first line (skip header)
    sum_norm = pd.DataFrame(0, index=range(block_size), columns=range(columns))  # Initialize the sum dataframe

    # Process each block until the file is finished
    while current_line + block_size <= len(lines):
        # Read the current block of 15979 lines and convert it to a DataFrame
        data_block = pd.DataFrame(
            [list(map(int, lines[current_line + i].split())) for i in range(block_size)],
            columns=range(columns)
        )

        # Row-normalize the block
        normalized_block = data_block.div(data_block.sum(axis=1), axis=0)

        # Handle undefined rows (all-zero rows become NaN during division)
        normalized_block = normalized_block.fillna(0)  # Replace NaN rows with zeros

        # Add to the cumulative sum
        sum_norm += normalized_block

        # Move to the next block (skip the next line and continue reading)
        current_line += block_size + 1

    # Final row-normalization of the cumulative sum
    sum_norm = sum_norm.div(sum_norm.sum(axis=1), axis=0)
    sum_norm = sum_norm.fillna(0)  # Replace any resulting NaN rows with zeros

    # Save to CSV
    sum_norm.to_csv(output_csv, index=False)
    print(f"Row-normalized cumulative sum saved to {output_csv}.")

# Example usage:
process_obj_contributions()