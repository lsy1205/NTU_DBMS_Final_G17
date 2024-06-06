import csv


def append_last_column(file1, file2):
    # Read the first CSV file
    with open(file1, "r") as f1:
        reader1 = csv.reader(f1)
        data1 = list(reader1)

    # Read the second CSV file
    with open(file2, "r") as f2:
        reader2 = csv.reader(f2)
        data2 = list(reader2)

    # Get the last column from the second file
    last_column = [row[-1] for row in data2]

    # Append the last column to the first file
    for i, row in enumerate(data1):
        row.append(last_column[i])

    # Write the updated data to a new file
    with open("OOD_test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data1)


# Usage example
file1 = "OOD_input_data.csv"
file2 = "OOD_output.csv"
append_last_column(file1, file2)
