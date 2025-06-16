DROP TABLE IF EXISTS iot23_raw;

CREATE TABLE iot23_raw (
    id SERIAL PRIMARY KEY,
    ts DOUBLE PRECISION,
    uid TEXT,
    id_orig_h TEXT,
    id_orig_p INTEGER,
    id_resp_h TEXT,
    id_resp_p INTEGER,
    proto TEXT,
    service TEXT,
    duration REAL,
    orig_bytes DOUBLE PRECISION,
    resp_bytes DOUBLE PRECISION,
    conn_state TEXT,
    local_orig BOOLEAN,
    local_resp BOOLEAN,
    missed_bytes DOUBLE PRECISION,
    history TEXT,
    orig_pkts INTEGER,
    orig_ip_bytes DOUBLE PRECISION,
    resp_pkts INTEGER,
    resp_ip_bytes DOUBLE PRECISION,
    tunnel_parents TEXT,
    label TEXT,
    detailed_label TEXT,
    source_file TEXT
);

SELECT *
FROM iot23_raw;