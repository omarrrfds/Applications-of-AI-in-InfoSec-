import os
import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------
# CONFIG
# ---------------------------
file_path = r"KDDTrain+.txt"
model_filename = "network_anomaly_detection_model.joblib"
upload_url = "http://10.129.105.42:8001/upload"  # <-- web form endpoint you specified

# NSL-KDD columns
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'
]

# ---------------------------
# LOAD DATA
# ---------------------------
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found: {os.path.abspath(file_path)}")

df = pd.read_csv(file_path, names=columns, header=None)
print("[+] Rows:", len(df))
print(df.head(3))

if len(df) < 1000:
    raise RuntimeError("Dataset too small — wrong file? (You should have ~125k rows for KDDTrain+.)")

# ---------------------------
# LABELS (your mapping)
# ---------------------------
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)

dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
# NOTE: your original code had 'loadmdoule' typo; keep it if you want EXACT, but it weakens mapping.
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail',
                  'snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

def map_attack(attack):
    if attack in dos_attacks:
        return 1
    elif attack in probe_attacks:
        return 2
    elif attack in privilege_attacks:
        return 3
    elif attack in access_attacks:
        return 4
    else:
        return 0

df['attack_map'] = df['attack'].apply(map_attack)

# ---------------------------
# FEATURES (your encoding)
# ---------------------------
features_to_encode = ['protocol_type', 'service']
encoded = pd.get_dummies(df[features_to_encode])

numeric_features = [
    'duration','src_bytes','dst_bytes','wrong_fragment','urgent','hot',
    'num_failed_logins','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

train_set = encoded.join(df[numeric_features])
multi_y = df['attack_map']

# ---------------------------
# SPLIT / TRAIN / EVAL
# ---------------------------
train_X, test_X, train_y, test_y = train_test_split(
    train_set, multi_y, test_size=0.2, random_state=1337
)

multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1337
)

rf_model_multi = RandomForestClassifier(random_state=1337, n_estimators=200, n_jobs=-1)
rf_model_multi.fit(multi_train_X, multi_train_y)

val_pred = rf_model_multi.predict(multi_val_X)
print("\nValidation Set Evaluation:")
print("Accuracy :", accuracy_score(multi_val_y, val_pred))
print("Precision:", precision_score(multi_val_y, val_pred, average='weighted', zero_division=0))
print("Recall   :", recall_score(multi_val_y, val_pred, average='weighted', zero_division=0))
print("F1-Score :", f1_score(multi_val_y, val_pred, average='weighted', zero_division=0))
print("\nClassification Report (Validation):")
print(classification_report(multi_val_y, val_pred,
      target_names=['Normal','DoS','Probe','Privilege','Access'], zero_division=0))

test_pred = rf_model_multi.predict(test_X)
print("\nTest Set Evaluation:")
print("Accuracy :", accuracy_score(test_y, test_pred))
print("Precision:", precision_score(test_y, test_pred, average='weighted', zero_division=0))
print("Recall   :", recall_score(test_y, test_pred, average='weighted', zero_division=0))
print("F1-Score :", f1_score(test_y, test_pred, average='weighted', zero_division=0))

# ---------------------------
# SAVE MODEL
# ---------------------------
joblib.dump(rf_model_multi, model_filename)
print(f"\n[+] Model saved -> {os.path.abspath(model_filename)}")

# ---------------------------
# UPLOAD MODEL
# ---------------------------
with open(model_filename, "rb") as f:
    r = requests.post(
        upload_url,
        files={"model": (model_filename, f, "application/octet-stream")},
        timeout=120
    )

print("\n==> Upload Status:", r.status_code)
print(r.text[:4000])  # likely HTML; flag may appear in this text
