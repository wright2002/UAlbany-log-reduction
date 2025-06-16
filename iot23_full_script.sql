
DROP TABLE IF EXISTS SessionHistory;
DROP TABLE IF EXISTS TunnelParent;
DROP TABLE IF EXISTS SessionService;
DROP TABLE IF EXISTS SessionLabel;
DROP TABLE IF EXISTS SessionDuration;
DROP TABLE IF EXISTS SessionOrigBytes;
DROP TABLE IF EXISTS SessionRespBytes;
DROP TABLE IF EXISTS SessionLocalOrig;
DROP TABLE IF EXISTS SessionLocalResp;
DROP TABLE IF EXISTS HistoryLegend;
DROP TABLE IF EXISTS Service;
DROP TABLE IF EXISTS Label;
DROP TABLE IF EXISTS Session;


CREATE TABLE Session (
    uid TEXT PRIMARY KEY,
    ts DOUBLE PRECISION,
    id_orig_h TEXT,
    id_orig_p INTEGER,
    id_resp_h TEXT,
    id_resp_p INTEGER,
    proto TEXT,
    conn_state TEXT,
    missed_bytes DOUBLE PRECISION,
    orig_pkts INTEGER,
    orig_ip_bytes DOUBLE PRECISION,
    resp_pkts INTEGER,
    resp_ip_bytes DOUBLE PRECISION,
    simple_label TEXT,
    source_file TEXT
);

CREATE TABLE SessionDuration (
    uid TEXT PRIMARY KEY,
    duration REAL,
    FOREIGN KEY (uid) REFERENCES Session(uid)
);

CREATE TABLE SessionOrigBytes (
    uid TEXT PRIMARY KEY,
    orig_bytes DOUBLE PRECISION,
    FOREIGN KEY (uid) REFERENCES Session(uid)
);

CREATE TABLE SessionRespBytes (
    uid TEXT PRIMARY KEY,
    resp_bytes DOUBLE PRECISION,
    FOREIGN KEY (uid) REFERENCES Session(uid)
);

CREATE TABLE SessionLocalOrig (
    uid TEXT PRIMARY KEY,
    local_orig BOOLEAN,
    FOREIGN KEY (uid) REFERENCES Session(uid)
);

CREATE TABLE SessionLocalResp (
    uid TEXT PRIMARY KEY,
    local_resp BOOLEAN,
    FOREIGN KEY (uid) REFERENCES Session(uid)
);

CREATE TABLE Service (
    svc_id SERIAL PRIMARY KEY,
    svc_name TEXT
);

CREATE TABLE SessionService (
    uid TEXT PRIMARY KEY,
    svc_id INTEGER,
    FOREIGN KEY (uid) REFERENCES Session(uid),
    FOREIGN KEY (svc_id) REFERENCES Service(svc_id)
);

CREATE TABLE Label (
    label_id SERIAL PRIMARY KEY,
    label_name TEXT
);

CREATE TABLE SessionLabel (
    uid TEXT PRIMARY KEY,
    label_id INTEGER,
    FOREIGN KEY (uid) REFERENCES Session(uid),
    FOREIGN KEY (label_id) REFERENCES Label(label_id)
);

CREATE TABLE TunnelParent (
    child_uid TEXT,
    parent_uid TEXT,
    PRIMARY KEY (child_uid, parent_uid),
    FOREIGN KEY (child_uid) REFERENCES Session(uid),
    FOREIGN KEY (parent_uid) REFERENCES Session(uid)
);

CREATE TABLE HistoryLegend (
    id SERIAL PRIMARY KEY,
    code CHAR(1),
    description TEXT
);

CREATE TABLE SessionHistory (
    uid TEXT,
    index INTEGER,
    history_id INTEGER,
    PRIMARY KEY (uid, index),
    FOREIGN KEY (uid) REFERENCES Session(uid),
    FOREIGN KEY (history_id) REFERENCES HistoryLegend(id)
);


DELETE FROM HistoryLegend;

INSERT INTO HistoryLegend (code, description) VALUES
('s', 'SYN seen on originator side'),
('S', 'SYN seen on responder side'),
('h', 'connection established from originator'),
('H', 'connection established from responder'),
('a', 'ACK seen from originator'),
('A', 'ACK seen from responder'),
('d', 'data seen from originator'),
('D', 'data seen from responder'),
('t', 'retransmissions from originator'),
('T', 'retransmissions from responder or truncated connection'),
('c', 'connection completed from originator'),
('C', 'connection completed from responder'),
('f', 'FIN seen from originator'),
('F', 'FIN seen from responder / connection attempt failed'),
('r', 'RST seen from originator'),
('R', 'RST seen from responder'),
('g', 'content gap'),
('^', 'connection originated here'),
('<', 'connection reversed direction');

DELETE FROM Service;

INSERT INTO Service (svc_name)
SELECT DISTINCT service
FROM iot23_raw
WHERE service IS NOT NULL;

DELETE FROM Label;

INSERT INTO Label (label_name)
SELECT DISTINCT detailed_label
FROM iot23_raw
WHERE detailed_label IS NOT NULL;

DELETE FROM Session;

INSERT INTO Session (
    uid, ts, id_orig_h, id_orig_p, id_resp_h, id_resp_p,
    proto, conn_state, missed_bytes, orig_pkts, orig_ip_bytes,
    resp_pkts, resp_ip_bytes, simple_label, source_file
)
SELECT
    uid, ts, id_orig_h, id_orig_p, id_resp_h, id_resp_p,
    proto, conn_state, missed_bytes, orig_pkts, orig_ip_bytes,
    resp_pkts, resp_ip_bytes,
    CASE
        WHEN label ILIKE 'benign' THEN 'Benign'
        ELSE label
    END AS simple_label,
    source_file
FROM iot23_raw;

DELETE FROM SessionDuration;

INSERT INTO SessionDuration (uid, duration)
SELECT uid, duration
FROM iot23_raw
WHERE duration IS NOT NULL;

DELETE FROM SessionOrigBytes;

INSERT INTO SessionOrigBytes (uid, orig_bytes)
SELECT uid, orig_bytes
FROM iot23_raw
WHERE orig_bytes IS NOT NULL;

DELETE FROM SessionRespBytes;

INSERT INTO SessionRespBytes (uid, resp_bytes)
SELECT uid, resp_bytes
FROM iot23_raw
WHERE resp_bytes IS NOT NULL;

DELETE FROM SessionLocalOrig;

INSERT INTO SessionLocalOrig (uid, local_orig)
SELECT uid, local_orig
FROM iot23_raw
WHERE local_orig IS NOT NULL;

DELETE FROM SessionLocalResp;

INSERT INTO SessionLocalResp (uid, local_resp)
SELECT uid, local_resp
FROM iot23_raw
WHERE local_resp IS NOT NULL;

DELETE FROM SessionService;

INSERT INTO SessionService (uid, svc_id)
SELECT i.uid, s.svc_id
FROM iot23_raw i
JOIN Service s ON i.service = s.svc_name
WHERE i.service IS NOT NULL;

DELETE FROM SessionLabel;

INSERT INTO SessionLabel (uid, label_id)
SELECT i.uid, l.label_id
FROM iot23_raw i
JOIN Label l ON i.label = l.label_name
WHERE i.label IS NOT NULL;

DELETE FROM TunnelParent;

INSERT INTO TunnelParent (child_uid, parent_uid)
SELECT uid, tunnel_parents
FROM iot23_raw
WHERE tunnel_parents IS NOT NULL;
