import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get DB credentials from env
username = os.getenv('PGUSER')
password = os.getenv('PGPASSWORD')
host = os.getenv('PGHOST', 'localhost')
port = os.getenv('PGPORT', '5432')
database = os.getenv('PGDATABASE')

# Create engine and query
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
query = "SELECT * FROM training_data_2;"
df = pd.read_sql_query(query, engine)

print(df.head())
print(df.shape)

df.to_parquet("training_data_2.parquet", index=False)
