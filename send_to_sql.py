import os
import subprocess
from pathlib import Path

def load_csv_to_postgres(csv_path, source_label, db_user, db_name, db_host='localhost', db_port=5432, table_name='iot23_raw', db_password=None):
    # Temporary CSV with source_file column
    tmp_path = csv_path.with_suffix('.tmp.csv')

    with open(csv_path, 'r', encoding='utf-8') as infile, open(tmp_path, 'w', encoding='utf-8') as outfile:
        header = infile.readline().strip()
        outfile.write(header + ',source_file\n')
        for line in infile:
            outfile.write(line.strip() + f',{source_label}\n')

    # Set password in environment
    env = os.environ.copy()
    if db_password:
        env['PGPASSWORD'] = db_password

    # Execute the psql \copy command
    psql_path = r"C:\Program Files\PostgreSQL\17\bin\psql.exe"

    cmd = [
        psql_path,
        '-U', db_user,
        '-d', db_name,
        '-h', db_host,
        '-p', str(db_port),
        '-c', f"\\copy {table_name} (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state, local_orig, local_resp, missed_bytes, history, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes, tunnel_parents, label, detailed_label, source_file) FROM '{tmp_path}' CSV HEADER"
    ]

    print(f"Importing: {csv_path.name}")
    subprocess.run(cmd, env=env, check=True)
    os.remove(tmp_path)

def main():
    directory = input("Enter directory containing CSV files: ").strip()
    db_host = input("PostgreSQL host (e.g., localhost or remote IP): ").strip()
    db_port = input("PostgreSQL port (default 5432): ").strip()
    db_port = int(db_port) if db_port else 5432
    db_user = input("PostgreSQL username: ").strip()
    db_name = input("Database name: ").strip()
    db_password = input("Database password: ").strip()

    csv_dir = Path(directory)
    csv_files = list(csv_dir.glob("*.csv"))

    for csv_file in csv_files:
        source_label = csv_file.name
        load_csv_to_postgres(
            csv_file, source_label,
            db_user=db_user,
            db_name=db_name,
            db_host=db_host,
            db_port=db_port,
            db_password=db_password
        )

    print("All CSVs imported successfully.")

if __name__ == "__main__":
    main()
