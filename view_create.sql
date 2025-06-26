DROP MATERIALIZED VIEW IF EXISTS session_ctu_malware_1_1_full;

CREATE MATERIALIZED VIEW session_ctu_malware_1_1_full AS
WITH history_array AS (
    SELECT
        sh.uid,
        ARRAY_AGG(hl.code ORDER BY sh.index) AS history_codes
    FROM SessionHistory sh
    JOIN HistoryLegend hl ON sh.history_id = hl.id
    GROUP BY sh.uid
)

SELECT
    s.*,

	-- IP address as 32-bit integers
	(CAST(split_part(s.id_orig_h, '.', 1) AS bigint) << 24) +
	(CAST(split_part(s.id_orig_h, '.', 2) AS bigint) << 16) +
	(CAST(split_part(s.id_orig_h, '.', 3) AS bigint) << 8) +
	CAST(split_part(s.id_orig_h, '.', 4) AS bigint) AS id_orig_h_int,

	(CAST(split_part(s.id_resp_h, '.', 1) AS bigint) << 24) +
	(CAST(split_part(s.id_resp_h, '.', 2) AS bigint) << 16) +
	(CAST(split_part(s.id_resp_h, '.', 3) AS bigint) << 8) +
	CAST(split_part(s.id_resp_h, '.', 4) AS bigint) AS id_resp_h_int,


	(EXTRACT(EPOCH FROM TO_TIMESTAMP(s.ts)) % 86400)::int AS time_of_day_seconds,

    -- Duration with missing indicator
    COALESCE(sd.duration, 0) AS duration,
    CASE WHEN sd.duration IS NULL THEN 1 ELSE 0 END AS duration_missing,

    -- Orig bytes with missing indicator
    COALESCE(so.orig_bytes, 0) AS orig_bytes,
    CASE WHEN so.orig_bytes IS NULL THEN 1 ELSE 0 END AS orig_bytes_missing,

    -- Resp bytes with missing indicator
    COALESCE(sr.resp_bytes, 0) AS resp_bytes,
    CASE WHEN sr.resp_bytes IS NULL THEN 1 ELSE 0 END AS resp_bytes_missing,

    -- Service name
    sv.svc_name,

	-- One-hot for service name
	CASE WHEN sv.svc_name = 'dhcp' THEN 1 ELSE 0 END AS svc_dhcp,
	CASE WHEN sv.svc_name = 'dns' THEN 1 ELSE 0 END AS svc_dns,
	CASE WHEN sv.svc_name = 'http' THEN 1 ELSE 0 END AS svc_http,
	CASE WHEN sv.svc_name = 'irc' THEN 1 ELSE 0 END AS svc_irc,
	CASE WHEN sv.svc_name = 'ssh' THEN 1 ELSE 0 END AS svc_ssh,
	CASE WHEN sv.svc_name = 'ssl' THEN 1 ELSE 0 END AS svc_ssl,


    -- Tunnel parent
    tp.parent_uid,

    -- History codes
	ha.history_codes,

	-- One hot for history
	CASE WHEN history_codes[1] = 'S' THEN 1 ELSE 0 END AS history_1_SS,
	CASE WHEN history_codes[1] = 'h' THEN 1 ELSE 0 END AS history_1_h,
	CASE WHEN history_codes[1] = 'H' THEN 1 ELSE 0 END AS history_1_HH,
	CASE WHEN history_codes[1] = 'a' THEN 1 ELSE 0 END AS history_1_a,
	CASE WHEN history_codes[1] = 'A' THEN 1 ELSE 0 END AS history_1_AA,
	CASE WHEN history_codes[1] = 'd' THEN 1 ELSE 0 END AS history_1_d,
	CASE WHEN history_codes[1] = 'D' THEN 1 ELSE 0 END AS history_1_DD,
	CASE WHEN history_codes[1] = 't' THEN 1 ELSE 0 END AS history_1_t,
	CASE WHEN history_codes[1] = 'T' THEN 1 ELSE 0 END AS history_1_TT,
	CASE WHEN history_codes[1] = 'c' THEN 1 ELSE 0 END AS history_1_c,
	CASE WHEN history_codes[1] = 'C' THEN 1 ELSE 0 END AS history_1_CC,
	CASE WHEN history_codes[1] = 'f' THEN 1 ELSE 0 END AS history_1_f,
	CASE WHEN history_codes[1] = 'F' THEN 1 ELSE 0 END AS history_1_FF,
	CASE WHEN history_codes[1] = 'r' THEN 1 ELSE 0 END AS history_1_r,
	CASE WHEN history_codes[1] = 'R' THEN 1 ELSE 0 END AS history_1_RR,
	CASE WHEN history_codes[1] = 'g' THEN 1 ELSE 0 END AS history_1_g,
	CASE WHEN history_codes[1] = '^' THEN 1 ELSE 0 END AS history_1_caret,
	CASE WHEN history_codes[1] = 'w' THEN 1 ELSE 0 END AS history_1_w,
	CASE WHEN history_codes[1] = 'W' THEN 1 ELSE 0 END AS history_1_WW,
	CASE WHEN history_codes[1] = 'G' THEN 1 ELSE 0 END AS history_1_GG,
	CASE WHEN history_codes[1] = 'I' THEN 1 ELSE 0 END AS history_1_II,

	CASE WHEN history_codes[2] = 'S' THEN 1 ELSE 0 END AS history_2_SS,
	CASE WHEN history_codes[2] = 'h' THEN 1 ELSE 0 END AS history_2_h,
	CASE WHEN history_codes[2] = 'H' THEN 1 ELSE 0 END AS history_2_HH,
	CASE WHEN history_codes[2] = 'a' THEN 1 ELSE 0 END AS history_2_a,
	CASE WHEN history_codes[2] = 'A' THEN 1 ELSE 0 END AS history_2_AA,
	CASE WHEN history_codes[2] = 'd' THEN 1 ELSE 0 END AS history_2_d,
	CASE WHEN history_codes[2] = 'D' THEN 1 ELSE 0 END AS history_2_DD,
	CASE WHEN history_codes[2] = 't' THEN 1 ELSE 0 END AS history_2_t,
	CASE WHEN history_codes[2] = 'T' THEN 1 ELSE 0 END AS history_2_TT,
	CASE WHEN history_codes[2] = 'c' THEN 1 ELSE 0 END AS history_2_c,
	CASE WHEN history_codes[2] = 'C' THEN 1 ELSE 0 END AS history_2_CC,
	CASE WHEN history_codes[2] = 'f' THEN 1 ELSE 0 END AS history_2_f,
	CASE WHEN history_codes[2] = 'F' THEN 1 ELSE 0 END AS history_2_FF,
	CASE WHEN history_codes[2] = 'r' THEN 1 ELSE 0 END AS history_2_r,
	CASE WHEN history_codes[2] = 'R' THEN 1 ELSE 0 END AS history_2_RR,
	CASE WHEN history_codes[2] = 'g' THEN 1 ELSE 0 END AS history_2_g,
	CASE WHEN history_codes[2] = '^' THEN 1 ELSE 0 END AS history_2_caret,
	CASE WHEN history_codes[2] = 'w' THEN 1 ELSE 0 END AS history_2_w,
	CASE WHEN history_codes[2] = 'W' THEN 1 ELSE 0 END AS history_2_WW,
	CASE WHEN history_codes[2] = 'G' THEN 1 ELSE 0 END AS history_2_GG,
	CASE WHEN history_codes[2] = 'I' THEN 1 ELSE 0 END AS history_2_II,

	CASE WHEN history_codes[3] = 'S' THEN 1 ELSE 0 END AS history_3_SS,
	CASE WHEN history_codes[3] = 'h' THEN 1 ELSE 0 END AS history_3_h,
	CASE WHEN history_codes[3] = 'H' THEN 1 ELSE 0 END AS history_3_HH,
	CASE WHEN history_codes[3] = 'a' THEN 1 ELSE 0 END AS history_3_a,
	CASE WHEN history_codes[3] = 'A' THEN 1 ELSE 0 END AS history_3_AA,
	CASE WHEN history_codes[3] = 'd' THEN 1 ELSE 0 END AS history_3_d,
	CASE WHEN history_codes[3] = 'D' THEN 1 ELSE 0 END AS history_3_DD,
	CASE WHEN history_codes[3] = 't' THEN 1 ELSE 0 END AS history_3_t,
	CASE WHEN history_codes[3] = 'T' THEN 1 ELSE 0 END AS history_3_TT,
	CASE WHEN history_codes[3] = 'c' THEN 1 ELSE 0 END AS history_3_c,
	CASE WHEN history_codes[3] = 'C' THEN 1 ELSE 0 END AS history_3_CC,
	CASE WHEN history_codes[3] = 'f' THEN 1 ELSE 0 END AS history_3_f,
	CASE WHEN history_codes[3] = 'F' THEN 1 ELSE 0 END AS history_3_FF,
	CASE WHEN history_codes[3] = 'r' THEN 1 ELSE 0 END AS history_3_r,
	CASE WHEN history_codes[3] = 'R' THEN 1 ELSE 0 END AS history_3_RR,
	CASE WHEN history_codes[3] = 'g' THEN 1 ELSE 0 END AS history_3_g,
	CASE WHEN history_codes[3] = '^' THEN 1 ELSE 0 END AS history_3_caret,
	CASE WHEN history_codes[3] = 'w' THEN 1 ELSE 0 END AS history_3_w,
	CASE WHEN history_codes[3] = 'W' THEN 1 ELSE 0 END AS history_3_WW,
	CASE WHEN history_codes[3] = 'G' THEN 1 ELSE 0 END AS history_3_GG,
	CASE WHEN history_codes[3] = 'I' THEN 1 ELSE 0 END AS history_3_II,

	CASE WHEN history_codes[4] = 'S' THEN 1 ELSE 0 END AS history_4_SS,
	CASE WHEN history_codes[4] = 'h' THEN 1 ELSE 0 END AS history_4_h,
	CASE WHEN history_codes[4] = 'H' THEN 1 ELSE 0 END AS history_4_HH,
	CASE WHEN history_codes[4] = 'a' THEN 1 ELSE 0 END AS history_4_a,
	CASE WHEN history_codes[4] = 'A' THEN 1 ELSE 0 END AS history_4_AA,
	CASE WHEN history_codes[4] = 'd' THEN 1 ELSE 0 END AS history_4_d,
	CASE WHEN history_codes[4] = 'D' THEN 1 ELSE 0 END AS history_4_DD,
	CASE WHEN history_codes[4] = 't' THEN 1 ELSE 0 END AS history_4_t,
	CASE WHEN history_codes[4] = 'T' THEN 1 ELSE 0 END AS history_4_TT,
	CASE WHEN history_codes[4] = 'c' THEN 1 ELSE 0 END AS history_4_c,
	CASE WHEN history_codes[4] = 'C' THEN 1 ELSE 0 END AS history_4_CC,
	CASE WHEN history_codes[4] = 'f' THEN 1 ELSE 0 END AS history_4_f,
	CASE WHEN history_codes[4] = 'F' THEN 1 ELSE 0 END AS history_4_FF,
	CASE WHEN history_codes[4] = 'r' THEN 1 ELSE 0 END AS history_4_r,
	CASE WHEN history_codes[4] = 'R' THEN 1 ELSE 0 END AS history_4_RR,
	CASE WHEN history_codes[4] = 'g' THEN 1 ELSE 0 END AS history_4_g,
	CASE WHEN history_codes[4] = '^' THEN 1 ELSE 0 END AS history_4_caret,
	CASE WHEN history_codes[4] = 'w' THEN 1 ELSE 0 END AS history_4_w,
	CASE WHEN history_codes[4] = 'W' THEN 1 ELSE 0 END AS history_4_WW,
	CASE WHEN history_codes[4] = 'G' THEN 1 ELSE 0 END AS history_4_GG,
	CASE WHEN history_codes[4] = 'I' THEN 1 ELSE 0 END AS history_4_II,

	CASE WHEN history_codes[5] = 'S' THEN 1 ELSE 0 END AS history_5_SS,
	CASE WHEN history_codes[5] = 'h' THEN 1 ELSE 0 END AS history_5_h,
	CASE WHEN history_codes[5] = 'H' THEN 1 ELSE 0 END AS history_5_HH,
	CASE WHEN history_codes[5] = 'a' THEN 1 ELSE 0 END AS history_5_a,
	CASE WHEN history_codes[5] = 'A' THEN 1 ELSE 0 END AS history_5_AA,
	CASE WHEN history_codes[5] = 'd' THEN 1 ELSE 0 END AS history_5_d,
	CASE WHEN history_codes[5] = 'D' THEN 1 ELSE 0 END AS history_5_DD,
	CASE WHEN history_codes[5] = 't' THEN 1 ELSE 0 END AS history_5_t,
	CASE WHEN history_codes[5] = 'T' THEN 1 ELSE 0 END AS history_5_TT,
	CASE WHEN history_codes[5] = 'c' THEN 1 ELSE 0 END AS history_5_c,
	CASE WHEN history_codes[5] = 'C' THEN 1 ELSE 0 END AS history_5_CC,
	CASE WHEN history_codes[5] = 'f' THEN 1 ELSE 0 END AS history_5_f,
	CASE WHEN history_codes[5] = 'F' THEN 1 ELSE 0 END AS history_5_FF,
	CASE WHEN history_codes[5] = 'r' THEN 1 ELSE 0 END AS history_5_r,
	CASE WHEN history_codes[5] = 'R' THEN 1 ELSE 0 END AS history_5_RR,
	CASE WHEN history_codes[5] = 'g' THEN 1 ELSE 0 END AS history_5_g,
	CASE WHEN history_codes[5] = '^' THEN 1 ELSE 0 END AS history_5_caret,
	CASE WHEN history_codes[5] = 'w' THEN 1 ELSE 0 END AS history_5_w,
	CASE WHEN history_codes[5] = 'W' THEN 1 ELSE 0 END AS history_5_WW,
	CASE WHEN history_codes[5] = 'G' THEN 1 ELSE 0 END AS history_5_GG,
	CASE WHEN history_codes[5] = 'I' THEN 1 ELSE 0 END AS history_5_II,

	CASE WHEN history_codes[6] = 'S' THEN 1 ELSE 0 END AS history_6_SS,
	CASE WHEN history_codes[6] = 'h' THEN 1 ELSE 0 END AS history_6_h,
	CASE WHEN history_codes[6] = 'H' THEN 1 ELSE 0 END AS history_6_HH,
	CASE WHEN history_codes[6] = 'a' THEN 1 ELSE 0 END AS history_6_a,
	CASE WHEN history_codes[6] = 'A' THEN 1 ELSE 0 END AS history_6_AA,
	CASE WHEN history_codes[6] = 'd' THEN 1 ELSE 0 END AS history_6_d,
	CASE WHEN history_codes[6] = 'D' THEN 1 ELSE 0 END AS history_6_DD,
	CASE WHEN history_codes[6] = 't' THEN 1 ELSE 0 END AS history_6_t,
	CASE WHEN history_codes[6] = 'T' THEN 1 ELSE 0 END AS history_6_TT,
	CASE WHEN history_codes[6] = 'c' THEN 1 ELSE 0 END AS history_6_c,
	CASE WHEN history_codes[6] = 'C' THEN 1 ELSE 0 END AS history_6_CC,
	CASE WHEN history_codes[6] = 'f' THEN 1 ELSE 0 END AS history_6_f,
	CASE WHEN history_codes[6] = 'F' THEN 1 ELSE 0 END AS history_6_FF,
	CASE WHEN history_codes[6] = 'r' THEN 1 ELSE 0 END AS history_6_r,
	CASE WHEN history_codes[6] = 'R' THEN 1 ELSE 0 END AS history_6_RR,
	CASE WHEN history_codes[6] = 'g' THEN 1 ELSE 0 END AS history_6_g,
	CASE WHEN history_codes[6] = '^' THEN 1 ELSE 0 END AS history_6_caret,
	CASE WHEN history_codes[6] = 'w' THEN 1 ELSE 0 END AS history_6_w,
	CASE WHEN history_codes[6] = 'W' THEN 1 ELSE 0 END AS history_6_WW,
	CASE WHEN history_codes[6] = 'G' THEN 1 ELSE 0 END AS history_6_GG,
	CASE WHEN history_codes[6] = 'I' THEN 1 ELSE 0 END AS history_6_II,

	CASE WHEN history_codes[7] = 'S' THEN 1 ELSE 0 END AS history_7_SS,
	CASE WHEN history_codes[7] = 'h' THEN 1 ELSE 0 END AS history_7_h,
	CASE WHEN history_codes[7] = 'H' THEN 1 ELSE 0 END AS history_7_HH,
	CASE WHEN history_codes[7] = 'a' THEN 1 ELSE 0 END AS history_7_a,
	CASE WHEN history_codes[7] = 'A' THEN 1 ELSE 0 END AS history_7_AA,
	CASE WHEN history_codes[7] = 'd' THEN 1 ELSE 0 END AS history_7_d,
	CASE WHEN history_codes[7] = 'D' THEN 1 ELSE 0 END AS history_7_DD,
	CASE WHEN history_codes[7] = 't' THEN 1 ELSE 0 END AS history_7_t,
	CASE WHEN history_codes[7] = 'T' THEN 1 ELSE 0 END AS history_7_TT,
	CASE WHEN history_codes[7] = 'c' THEN 1 ELSE 0 END AS history_7_c,
	CASE WHEN history_codes[7] = 'C' THEN 1 ELSE 0 END AS history_7_CC,
	CASE WHEN history_codes[7] = 'f' THEN 1 ELSE 0 END AS history_7_f,
	CASE WHEN history_codes[7] = 'F' THEN 1 ELSE 0 END AS history_7_FF,
	CASE WHEN history_codes[7] = 'r' THEN 1 ELSE 0 END AS history_7_r,
	CASE WHEN history_codes[7] = 'R' THEN 1 ELSE 0 END AS history_7_RR,
	CASE WHEN history_codes[7] = 'g' THEN 1 ELSE 0 END AS history_7_g,
	CASE WHEN history_codes[7] = '^' THEN 1 ELSE 0 END AS history_7_caret,
	CASE WHEN history_codes[7] = 'w' THEN 1 ELSE 0 END AS history_7_w,
	CASE WHEN history_codes[7] = 'W' THEN 1 ELSE 0 END AS history_7_WW,
	CASE WHEN history_codes[7] = 'G' THEN 1 ELSE 0 END AS history_7_GG,
	CASE WHEN history_codes[7] = 'I' THEN 1 ELSE 0 END AS history_7_II,

	CASE WHEN history_codes[8] = 'S' THEN 1 ELSE 0 END AS history_8_SS,
	CASE WHEN history_codes[8] = 'h' THEN 1 ELSE 0 END AS history_8_h,
	CASE WHEN history_codes[8] = 'H' THEN 1 ELSE 0 END AS history_8_HH,
	CASE WHEN history_codes[8] = 'a' THEN 1 ELSE 0 END AS history_8_a,
	CASE WHEN history_codes[8] = 'A' THEN 1 ELSE 0 END AS history_8_AA,
	CASE WHEN history_codes[8] = 'd' THEN 1 ELSE 0 END AS history_8_d,
	CASE WHEN history_codes[8] = 'D' THEN 1 ELSE 0 END AS history_8_DD,
	CASE WHEN history_codes[8] = 't' THEN 1 ELSE 0 END AS history_8_t,
	CASE WHEN history_codes[8] = 'T' THEN 1 ELSE 0 END AS history_8_TT,
	CASE WHEN history_codes[8] = 'c' THEN 1 ELSE 0 END AS history_8_c,
	CASE WHEN history_codes[8] = 'C' THEN 1 ELSE 0 END AS history_8_CC,
	CASE WHEN history_codes[8] = 'f' THEN 1 ELSE 0 END AS history_8_f,
	CASE WHEN history_codes[8] = 'F' THEN 1 ELSE 0 END AS history_8_FF,
	CASE WHEN history_codes[8] = 'r' THEN 1 ELSE 0 END AS history_8_r,
	CASE WHEN history_codes[8] = 'R' THEN 1 ELSE 0 END AS history_8_RR,
	CASE WHEN history_codes[8] = 'g' THEN 1 ELSE 0 END AS history_8_g,
	CASE WHEN history_codes[8] = '^' THEN 1 ELSE 0 END AS history_8_caret,
	CASE WHEN history_codes[8] = 'w' THEN 1 ELSE 0 END AS history_8_w,
	CASE WHEN history_codes[8] = 'W' THEN 1 ELSE 0 END AS history_8_WW,
	CASE WHEN history_codes[8] = 'G' THEN 1 ELSE 0 END AS history_8_GG,
	CASE WHEN history_codes[8] = 'I' THEN 1 ELSE 0 END AS history_8_II,

	CASE WHEN history_codes[9] = 'S' THEN 1 ELSE 0 END AS history_9_SS,
	CASE WHEN history_codes[9] = 'h' THEN 1 ELSE 0 END AS history_9_h,
	CASE WHEN history_codes[9] = 'H' THEN 1 ELSE 0 END AS history_9_HH,
	CASE WHEN history_codes[9] = 'a' THEN 1 ELSE 0 END AS history_9_a,
	CASE WHEN history_codes[9] = 'A' THEN 1 ELSE 0 END AS history_9_AA,
	CASE WHEN history_codes[9] = 'd' THEN 1 ELSE 0 END AS history_9_d,
	CASE WHEN history_codes[9] = 'D' THEN 1 ELSE 0 END AS history_9_DD,
	CASE WHEN history_codes[9] = 't' THEN 1 ELSE 0 END AS history_9_t,
	CASE WHEN history_codes[9] = 'T' THEN 1 ELSE 0 END AS history_9_TT,
	CASE WHEN history_codes[9] = 'c' THEN 1 ELSE 0 END AS history_9_c,
	CASE WHEN history_codes[9] = 'C' THEN 1 ELSE 0 END AS history_9_CC,
	CASE WHEN history_codes[9] = 'f' THEN 1 ELSE 0 END AS history_9_f,
	CASE WHEN history_codes[9] = 'F' THEN 1 ELSE 0 END AS history_9_FF,
	CASE WHEN history_codes[9] = 'r' THEN 1 ELSE 0 END AS history_9_r,
	CASE WHEN history_codes[9] = 'R' THEN 1 ELSE 0 END AS history_9_RR,
	CASE WHEN history_codes[9] = 'g' THEN 1 ELSE 0 END AS history_9_g,
	CASE WHEN history_codes[9] = '^' THEN 1 ELSE 0 END AS history_9_caret,
	CASE WHEN history_codes[9] = 'w' THEN 1 ELSE 0 END AS history_9_w,
	CASE WHEN history_codes[9] = 'W' THEN 1 ELSE 0 END AS history_9_WW,
	CASE WHEN history_codes[9] = 'G' THEN 1 ELSE 0 END AS history_9_GG,
	CASE WHEN history_codes[9] = 'I' THEN 1 ELSE 0 END AS history_9_II,

	CASE WHEN history_codes[10] = 'S' THEN 1 ELSE 0 END AS history_10_SS,
	CASE WHEN history_codes[10] = 'h' THEN 1 ELSE 0 END AS history_10_h,
	CASE WHEN history_codes[10] = 'H' THEN 1 ELSE 0 END AS history_10_HH,
	CASE WHEN history_codes[10] = 'a' THEN 1 ELSE 0 END AS history_10_a,
	CASE WHEN history_codes[10] = 'A' THEN 1 ELSE 0 END AS history_10_AA,
	CASE WHEN history_codes[10] = 'd' THEN 1 ELSE 0 END AS history_10_d,
	CASE WHEN history_codes[10] = 'D' THEN 1 ELSE 0 END AS history_10_DD,
	CASE WHEN history_codes[10] = 't' THEN 1 ELSE 0 END AS history_10_t,
	CASE WHEN history_codes[10] = 'T' THEN 1 ELSE 0 END AS history_10_TT,
	CASE WHEN history_codes[10] = 'c' THEN 1 ELSE 0 END AS history_10_c,
	CASE WHEN history_codes[10] = 'C' THEN 1 ELSE 0 END AS history_10_CC,
	CASE WHEN history_codes[10] = 'f' THEN 1 ELSE 0 END AS history_10_f,
	CASE WHEN history_codes[10] = 'F' THEN 1 ELSE 0 END AS history_10_FF,
	CASE WHEN history_codes[10] = 'r' THEN 1 ELSE 0 END AS history_10_r,
	CASE WHEN history_codes[10] = 'R' THEN 1 ELSE 0 END AS history_10_RR,
	CASE WHEN history_codes[10] = 'g' THEN 1 ELSE 0 END AS history_10_g,
	CASE WHEN history_codes[10] = '^' THEN 1 ELSE 0 END AS history_10_caret,
	CASE WHEN history_codes[10] = 'w' THEN 1 ELSE 0 END AS history_10_w,
	CASE WHEN history_codes[10] = 'W' THEN 1 ELSE 0 END AS history_10_WW,
	CASE WHEN history_codes[10] = 'G' THEN 1 ELSE 0 END AS history_10_GG,
	CASE WHEN history_codes[10] = 'I' THEN 1 ELSE 0 END AS history_10_II,

	CASE WHEN history_codes[11] = 'S' THEN 1 ELSE 0 END AS history_11_SS,
	CASE WHEN history_codes[11] = 'h' THEN 1 ELSE 0 END AS history_11_h,
	CASE WHEN history_codes[11] = 'H' THEN 1 ELSE 0 END AS history_11_HH,
	CASE WHEN history_codes[11] = 'a' THEN 1 ELSE 0 END AS history_11_a,
	CASE WHEN history_codes[11] = 'A' THEN 1 ELSE 0 END AS history_11_AA,
	CASE WHEN history_codes[11] = 'd' THEN 1 ELSE 0 END AS history_11_d,
	CASE WHEN history_codes[11] = 'D' THEN 1 ELSE 0 END AS history_11_DD,
	CASE WHEN history_codes[11] = 't' THEN 1 ELSE 0 END AS history_11_t,
	CASE WHEN history_codes[11] = 'T' THEN 1 ELSE 0 END AS history_11_TT,
	CASE WHEN history_codes[11] = 'c' THEN 1 ELSE 0 END AS history_11_c,
	CASE WHEN history_codes[11] = 'C' THEN 1 ELSE 0 END AS history_11_CC,
	CASE WHEN history_codes[11] = 'f' THEN 1 ELSE 0 END AS history_11_f,
	CASE WHEN history_codes[11] = 'F' THEN 1 ELSE 0 END AS history_11_FF,
	CASE WHEN history_codes[11] = 'r' THEN 1 ELSE 0 END AS history_11_r,
	CASE WHEN history_codes[11] = 'R' THEN 1 ELSE 0 END AS history_11_RR,
	CASE WHEN history_codes[11] = 'g' THEN 1 ELSE 0 END AS history_11_g,
	CASE WHEN history_codes[11] = '^' THEN 1 ELSE 0 END AS history_11_caret,
	CASE WHEN history_codes[11] = 'w' THEN 1 ELSE 0 END AS history_11_w,
	CASE WHEN history_codes[11] = 'W' THEN 1 ELSE 0 END AS history_11_WW,
	CASE WHEN history_codes[11] = 'G' THEN 1 ELSE 0 END AS history_11_GG,
	CASE WHEN history_codes[11] = 'I' THEN 1 ELSE 0 END AS history_11_II,

	CASE WHEN history_codes[12] = 'S' THEN 1 ELSE 0 END AS history_12_SS,
	CASE WHEN history_codes[12] = 'h' THEN 1 ELSE 0 END AS history_12_h,
	CASE WHEN history_codes[12] = 'H' THEN 1 ELSE 0 END AS history_12_HH,
	CASE WHEN history_codes[12] = 'a' THEN 1 ELSE 0 END AS history_12_a,
	CASE WHEN history_codes[12] = 'A' THEN 1 ELSE 0 END AS history_12_AA,
	CASE WHEN history_codes[12] = 'd' THEN 1 ELSE 0 END AS history_12_d,
	CASE WHEN history_codes[12] = 'D' THEN 1 ELSE 0 END AS history_12_DD,
	CASE WHEN history_codes[12] = 't' THEN 1 ELSE 0 END AS history_12_t,
	CASE WHEN history_codes[12] = 'T' THEN 1 ELSE 0 END AS history_12_TT,
	CASE WHEN history_codes[12] = 'c' THEN 1 ELSE 0 END AS history_12_c,
	CASE WHEN history_codes[12] = 'C' THEN 1 ELSE 0 END AS history_12_CC,
	CASE WHEN history_codes[12] = 'f' THEN 1 ELSE 0 END AS history_12_f,
	CASE WHEN history_codes[12] = 'F' THEN 1 ELSE 0 END AS history_12_FF,
	CASE WHEN history_codes[12] = 'r' THEN 1 ELSE 0 END AS history_12_r,
	CASE WHEN history_codes[12] = 'R' THEN 1 ELSE 0 END AS history_12_RR,
	CASE WHEN history_codes[12] = 'g' THEN 1 ELSE 0 END AS history_12_g,
	CASE WHEN history_codes[12] = '^' THEN 1 ELSE 0 END AS history_12_caret,
	CASE WHEN history_codes[12] = 'w' THEN 1 ELSE 0 END AS history_12_w,
	CASE WHEN history_codes[12] = 'W' THEN 1 ELSE 0 END AS history_12_WW,
	CASE WHEN history_codes[12] = 'G' THEN 1 ELSE 0 END AS history_12_GG,
	CASE WHEN history_codes[12] = 'I' THEN 1 ELSE 0 END AS history_12_II,


    -- One-hot for conn_state
    CASE WHEN s.conn_state = 'S0' THEN 1 ELSE 0 END AS conn_S0,
    CASE WHEN s.conn_state = 'S1' THEN 1 ELSE 0 END AS conn_S1,
    CASE WHEN s.conn_state = 'SF' THEN 1 ELSE 0 END AS conn_SF,
    CASE WHEN s.conn_state = 'REJ' THEN 1 ELSE 0 END AS conn_REJ,
    CASE WHEN s.conn_state = 'S2' THEN 1 ELSE 0 END AS conn_S2,
    CASE WHEN s.conn_state = 'S3' THEN 1 ELSE 0 END AS conn_S3,
    CASE WHEN s.conn_state = 'RSTO' THEN 1 ELSE 0 END AS conn_RSTO,
    CASE WHEN s.conn_state = 'RSTR' THEN 1 ELSE 0 END AS conn_RSTR,
    CASE WHEN s.conn_state = 'RSTOS0' THEN 1 ELSE 0 END AS conn_RSTOS0,
    CASE WHEN s.conn_state = 'RSTRH' THEN 1 ELSE 0 END AS conn_RSTRH,
    CASE WHEN s.conn_state = 'SH' THEN 1 ELSE 0 END AS conn_SH,
    CASE WHEN s.conn_state = 'SHR' THEN 1 ELSE 0 END AS conn_SHR,
    CASE WHEN s.conn_state = 'OTH' THEN 1 ELSE 0 END AS conn_OTH,

    -- One-hot for proto
    CASE WHEN s.proto = 'icmp' THEN 1 ELSE 0 END AS proto_icmp,
    CASE WHEN s.proto = 'tcp' THEN 1 ELSE 0 END AS proto_tcp,
    CASE WHEN s.proto = 'udp' THEN 1 ELSE 0 END AS proto_udp,

	-- One-hot bucketing for origin port
	CASE WHEN id_orig_p <= 1023 THEN 1 ELSE 0 END AS orig_port_well_known,
	CASE WHEN id_orig_p > 1023 AND id_orig_p <= 49151 THEN 1 ELSE 0 END AS orig_port_registered,
	CASE WHEN id_orig_p > 49151 THEN 1 ELSE 0 END AS orig_port_ephemeral,

	-- One-hot for destination port
	CASE WHEN id_resp_p = 22 THEN 1 ELSE 0 END AS resp_port_22,
	CASE WHEN id_resp_p = 23 THEN 1 ELSE 0 END AS resp_port_23,
	CASE WHEN id_resp_p = 80 THEN 1 ELSE 0 END AS resp_port_80,
	CASE WHEN id_resp_p = 81 THEN 1 ELSE 0 END AS resp_port_81,
	CASE WHEN id_resp_p = 443 THEN 1 ELSE 0 END AS resp_port_443,
	CASE WHEN id_resp_p = 8080 THEN 1 ELSE 0 END AS resp_port_8080,
	CASE WHEN id_resp_p = 8081 THEN 1 ELSE 0 END AS resp_port_8081,
	CASE WHEN id_resp_p = 2323 THEN 1 ELSE 0 END AS resp_port_2323,
	CASE WHEN id_resp_p = 992 THEN 1 ELSE 0 END AS resp_port_992,
	CASE WHEN id_resp_p = 37215 THEN 1 ELSE 0 END AS resp_port_37215,
	CASE WHEN id_resp_p = 52869 THEN 1 ELSE 0 END AS resp_port_52869,
	CASE 
	  WHEN id_resp_p IN (22, 23, 80, 81, 443, 8080, 8081, 2323, 992, 37215, 52869) THEN 0
	  ELSE 1 
	END AS resp_port_other,

	-- label encoding for simple_label
	CASE WHEN simple_label = 'Malicious' THEN 1 ELSE 0 END AS label

FROM session_ctu_malware_1_1 s
LEFT JOIN history_array ha ON s.uid = ha.uid
LEFT JOIN SessionDuration sd ON s.uid = sd.uid
LEFT JOIN SessionOrigBytes so ON s.uid = so.uid
LEFT JOIN SessionRespBytes sr ON s.uid = sr.uid
LEFT JOIN SessionService ss ON s.uid = ss.uid
LEFT JOIN Service sv ON ss.svc_id = sv.svc_id
LEFT JOIN TunnelParent tp ON s.uid = tp.child_uid;


SELECT *
FROM session_ctu_malware_1_1_full
LIMIT 10;
