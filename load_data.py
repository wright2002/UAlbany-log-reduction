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
    columns = [
        'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
        'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
        'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
        'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
        'resp_ip_bytes', 'tunnel_parents', 'label', 'detailed-label'
    ]

    df = pd.read_csv(
        filepath,
        sep='\t',
        names=columns,
        comment='#',
        parse_dates=['ts'],
        na_values=["-", "(empty)"],
        chunksize=2_000_000
    )
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
    print(filepath)
    clean_path = "conn_cleaned.log"

    preprocess_zeek_labels(filepath, clean_path)
    print("Preprocessing done.")

    chunk_iter = load_zeek_log(clean_path)
    first_chunk = next(chunk_iter)
    print("First chunk loaded. Shape:", first_chunk.shape)

    missing_values = check_missing_values(first_chunk)
    if not missing_values.empty:
        print("\nMissing Values:")
        print(missing_values)
    else:
        print("\nNo missing values found.")

    print("\nFirst full row:")
    print_row_inline(first_chunk, 0)

    print("\nSecond full row:")
    print_row_inline(first_chunk, 1)

    export = input("\nWould you like to export all chunks to a CSV file? (y/n): ").strip().lower()
    if export == 'y':
        outname = input("Enter output CSV filename (e.g., output.csv): ").strip()
        print(f"\nSaving cleaned DataFrame to '{outname}'...")

        # Re-load the iterator from the start
        chunk_iter = load_zeek_log(clean_path)
        for i, chunk in enumerate(chunk_iter):
            chunk.to_csv(outname, mode='a', header=(i == 0), index=False)

        print(f"All chunks exported to {outname}")


if __name__ == "__main__":
    main()