import pandas as pd
from io import StringIO

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Fix malformed label fields in data lines (space-separated labels at end)
def preprocess_zeek_labels(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        lines = []
        for line in infile:
            if line.startswith('#'):
                lines.append(line)
                continue

            parts = line.rstrip('\n').split('\t')
            if len(parts) == 21:
                split_labels = parts[-1].strip().split()
                if len(split_labels) == 3:
                    parts = parts[:-1] + split_labels

            lines.append('\t'.join(parts) + '\n')

        outfile.writelines(lines)


# Load cleaned Zeek log using parsed header and fixed label fields
def load_zeek_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    columns = []
    data_lines = []

    for line in lines:
        if line.startswith('#fields'):
            raw_fields = line.strip().split('\t')[1:]  # remove '#fields'
            fixed = []
            for part in raw_fields:
                if ' ' in part:
                    fixed.extend(part.strip().split())
                else:
                    fixed.append(part)
            columns = fixed
        elif not line.startswith('#'):
            data_lines.append(line.rstrip('\n'))

    data_str = '\n'.join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep='\t', names=columns, na_values=["-", "(empty)"], low_memory=False)
    return df

# Check for missing values
def check_missing_values(df):
    missing = df.isnull().sum()
    return missing[missing > 0]

# Print a row as a single line of col=value pairs
def print_row_inline(df, row_index):
    row = df.iloc[row_index]
    print(', '.join(f"{col}={row[col]}" for col in df.columns))

# Main execution block
def main():
    filepath = input('Enter file path: ')
    clean_path = "conn_cleaned.log"

    preprocess_zeek_labels(filepath, clean_path)
    print("Preprocessing done.")
    df = load_zeek_log(clean_path)

    print("DataFrame Loaded. Shape:", df.shape)

    missing_values = check_missing_values(df)
    if not missing_values.empty:
        print("\nMissing Values:")
        print(missing_values)
    else:
        print("\nNo missing values found.")

    print("\nFirst full row:")
    print_row_inline(df, 0)

    print("\nSecond full row:")
    print_row_inline(df, 1)

    export = input("\nWould you like to export the cleaned DataFrame to CSV? (y/n): ").strip().lower()
    if export == 'y':
        outname = input("Enter output CSV filename (e.g., output.csv): ").strip()
        print("\nSaving cleaned DataFrame to '{}'...".format(outname))
        df.to_csv(outname, index=False)
        print(f"DataFrame exported to {outname}")

if __name__ == "__main__":
    main()
