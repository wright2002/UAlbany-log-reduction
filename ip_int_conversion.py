import ipaddress

# Encode IPs as integers
df['id_orig_h_enc'] = df['id_orig_h'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))
df['id_resp_h_enc'] = df['id_resp_h'].apply(lambda ip: int(ipaddress.IPv4Address(ip)))

# Drop original IP columns
df.drop(columns=['id_orig_h', 'id_resp_h'], inplace=True)

#Check
print(df[['id_orig_h_enc', 'id_resp_h_enc']].head())