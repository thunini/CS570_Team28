

"""
Complete SW Smartwatch Dataset Pipeline: Graph Construction + Activity Recognition + Anomaly Detection
Adapted for smartwatch sensor data with motion, location, and activity labels
"""


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import timedelta, datetime
from collections import Counter, defaultdict
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GAE, VGAE, global_max_pool, LayerNorm, Set2Set
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 



warnings.filterwarnings('ignore')


from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import negative_sampling


# Set device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# =============================================================================
# STEP 1: SW DATASET LOADING AND PREPROCESSING (FIXED)
# =============================================================================

def load_sw_dataset(data_dir='data/CASAS_smart_watch_parquet/', num_files=49):
    """
    Fast loader for the CASAS SW smartwatch dataset using Parquet files.
    Randomly loads a specified number of Parquet files.
    Assumes each file contains 'timestamp' and 'user_activity_label' columns.
    """
    print(f"📦 Loading SW smartwatch dataset from Parquet files in {data_dir}...")

    all_files = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.parquet')]

    if len(all_files) < num_files:
        raise ValueError(f"Not enough Parquet files to sample {num_files}. Found only {len(all_files)}.")

    file_paths = random.sample(all_files, num_files)

    all_data = []
    files_loaded = 0

    for filepath in tqdm(file_paths, desc="📂 Loading Parquet files", ncols=100):
        try:
            df = pd.read_parquet(filepath)
            print(f"📄 {os.path.basename(filepath)}: {len(df)} rows")

            # Check required columns
            if 'timestamp' not in df.columns:
                print(f"⚠️ Missing 'timestamp' in {filepath}")
                continue
            if 'user_activity_label' not in df.columns:
                print(f"⚠️ Missing 'user_activity_label' in {filepath}")
                continue

            # Drop rows with missing activity labels
            df['user_activity_label'] = df['user_activity_label'].astype(str).str.strip()
            df['user_activity_label'].replace(['nan', 'None', 'null', ''], pd.NA, inplace=True)
            df = df[df['user_activity_label'].notna()].copy()

            if df.empty:
                print(f"⚠️ No valid activity rows in {filepath}")
                continue

            # Add participant ID
            participant_id = os.path.basename(filepath).split('.')[1].zfill(2)
            df['participant'] = participant_id

            all_data.append(df)
            files_loaded += 1

        except Exception as e:
            print(f"❌ Failed to load {filepath}: {e}")

    if not all_data:
        raise ValueError(f"No valid Parquet files found in {data_dir}")

    df_all = pd.concat(all_data, ignore_index=True)
    df_all.sort_values(by=['participant', 'timestamp'], inplace=True)

    print(f"\n✅ Loaded {files_loaded} files, {len(df_all)} labeled sensor events")
    print(f"👥 Participants: {df_all['participant'].nunique()}")
    print(f"📊 Activities: {df_all['user_activity_label'].value_counts().to_dict()}")

    return df_all



def create_smartwatch_windows(df_all, window_size=15, stride=5):
   """
   Create sliding windows for smartwatch temporal graphs (FIXED: smaller windows)
   """
   print(f"Creating smartwatch sliding windows (size={window_size}, stride={stride})...")


   windowed_samples = []
   grouped = df_all.groupby(['participant', 'user_activity_label'])


   for (participant, activity), group in grouped:
       if pd.isna(activity) or activity == 'nan':
           continue


       group = group.sort_values('timestamp').reset_index(drop=True)


       # FIXED: Ensure minimum data for meaningful windows
       if len(group) < window_size:
           print(f"  Skipping {participant}-{activity}: only {len(group)} samples (need {window_size})")
           continue


       max_start = len(group) - window_size + 1


       for start_idx in range(0, max_start, stride):
           end_idx = start_idx + window_size
           window_df = group.iloc[start_idx:end_idx].copy()


           # FIXED: Validate window has consistent activity and sufficient data
           if len(window_df) == window_size and window_df['user_activity_label'].nunique() == 1:
               windowed_samples.append({
                   'participant': participant,
                   'activity': activity,
                   'sensor_readings': window_df,
                   'start_time': window_df.iloc[0]['timestamp'],
                   'end_time': window_df.iloc[-1]['timestamp']
               })


   print(f"✅ Created {len(windowed_samples)} windowed samples")


   # FIXED: Print distribution to verify data quality
   if windowed_samples:
       activity_dist = Counter([sample['activity'] for sample in windowed_samples])
       print(f"📊 Window distribution: {dict(activity_dist)}")


   return windowed_samples


def build_smartwatch_mappings(df_all):
   """
   Build mappings for smartwatch sensors and activities (FIXED)
   """
   # Get unique activities (excluding NaN) - FIXED: better filtering
   activities = []
   for act in df_all['user_activity_label'].unique():
       if pd.notna(act) and str(act).strip() not in ['nan', 'None', 'null', '']:
           activities.append(str(act).strip())


   activities = sorted(list(set(activities)))
   activity_to_index = {activity: idx for idx, activity in enumerate(activities)}
   num_activities = len(activity_to_index)


   # FIXED: Define sensor feature columns based on actual SW dataset structure
   motion_sensors = ['yaw', 'pitch', 'roll', 'rotation_rate_x', 'rotation_rate_y', 'rotation_rate_z',
                     'user_acceleration_x', 'user_acceleration_y', 'user_acceleration_z']


   location_sensors = ['latitude_distance_from_mean', 'longitude_distance_from_mean',
                       'altitude_distance_from_mean', 'course', 'speed',
                       'horizontal_accuracy', 'vertical_accuracy']


   # Check which columns actually exist in the data
   available_motion = [col for col in motion_sensors if col in df_all.columns]
   available_location = [col for col in location_sensors if col in df_all.columns]


   print(f"📡 Available motion sensors: {available_motion}")
   print(f"📍 Available location sensors: {available_location}")


   # Virtual sensor nodes: Motion, Location, Battery, Context
   sensor_types = {
       'motion': 'motion_sensor',
       'location': 'location_sensor',
       'battery': 'battery_sensor',
       'context': 'context_sensor'
   }


   sensor_to_index = {sensor: idx for idx, sensor in enumerate(sensor_types.keys())}
   num_sensors = len(sensor_to_index)


   print(f"📡 Virtual sensors: {num_sensors} types")
   print(f"🔧 Motion features: {len(available_motion)}")
   print(f"📍 Location features: {len(available_location)}")
   print(f"🎯 Activities: {num_activities} classes - {activities}")


   return sensor_to_index, activity_to_index, available_motion, available_location, num_sensors, num_activities




def build_smartwatch_features(sensor_readings, motion_sensors, location_sensors, num_sensors):
   """
   Build feature matrix for smartwatch virtual sensor nodes (FIXED: robust feature extraction)
   """
   features = torch.zeros((num_sensors, 8), dtype=torch.float32)


   # FIXED: Motion sensor features (node 0) with error handling
   motion_data = []
   for col in motion_sensors:
       if col in sensor_readings.columns:
           values = pd.to_numeric(sensor_readings[col], errors='coerce').dropna()
           if len(values) > 0:
               motion_data.extend(values.tolist())


   if motion_data and len(motion_data) > 0:
       motion_array = np.array(motion_data)
       # Avoid division by zero and handle edge cases
       features[0, 0] = float(np.mean(motion_array))
       features[0, 1] = float(np.std(motion_array)) if len(motion_array) > 1 else 0.0
       features[0, 2] = float(np.min(motion_array))
       features[0, 3] = float(np.max(motion_array))
       features[0, 4] = float(len(motion_data))
       features[0, 5] = float(np.median(motion_array))
       features[0, 6] = float(np.percentile(motion_array, 25)) if len(motion_array) > 0 else 0.0
       features[0, 7] = float(np.percentile(motion_array, 75)) if len(motion_array) > 0 else 0.0


   # FIXED: Location sensor features (node 1) with error handling
   location_data = []
   for col in location_sensors:
       if col in sensor_readings.columns:
           values = pd.to_numeric(sensor_readings[col], errors='coerce').dropna()
           if len(values) > 0:
               location_data.extend(values.tolist())


   if location_data and len(location_data) > 0:
       location_array = np.array(location_data)
       features[1, 0] = float(np.mean(location_array))
       features[1, 1] = float(np.std(location_array)) if len(location_array) > 1 else 0.0
       features[1, 2] = float(np.min(location_array))
       features[1, 3] = float(np.max(location_array))
       features[1, 4] = float(len(location_data))
       features[1, 5] = float(np.median(location_array))
       features[1, 6] = float(np.percentile(location_array, 25)) if len(location_array) > 0 else 0.0
       features[1, 7] = float(np.percentile(location_array, 75)) if len(location_array) > 0 else 0.0


   # FIXED: Battery sensor features (node 2) with error handling
   if 'battery_state' in sensor_readings.columns:
       battery_data = sensor_readings['battery_state'].dropna()
       if len(battery_data) > 0:
           battery_states = battery_data.value_counts()
           total_readings = len(battery_data)


           features[2, 0] = float(battery_states.get('unplugged', 0))
           features[2, 1] = float(battery_states.get('charging', 0))
           features[2, 2] = float(battery_states.get('full', 0))
           features[2, 3] = float(total_readings)
           features[2, 4] = float(battery_states.get('unplugged', 0) / total_readings) if total_readings > 0 else 0.0
           features[2, 5] = float(battery_states.get('charging', 0) / total_readings) if total_readings > 0 else 0.0
           features[2, 6] = float(len(battery_states))
           features[2, 7] = 0.0


   # FIXED: Context sensor features (node 3) with error handling
   if 'timestamp' in sensor_readings.columns:
       timestamps = pd.to_datetime(sensor_readings['timestamp'], errors='coerce').dropna()
       if len(timestamps) > 0:
           time_span = (timestamps.max() - timestamps.min()).total_seconds()
           hour_of_day = timestamps.dt.hour.mean()
           day_of_week = timestamps.dt.dayofweek.mean()


           features[3, 0] = float(time_span)
           features[3, 1] = float(hour_of_day)
           features[3, 2] = float(day_of_week)
           features[3, 3] = float(len(timestamps))
           features[3, 4] = float(timestamps.dt.minute.std()) if len(timestamps) > 1 else 0.0


           if 'location_type' in sensor_readings.columns:
               location_types = sensor_readings['location_type'].dropna()
               if len(location_types) > 0:
                   location_type_counts = location_types.value_counts()
                   features[3, 5] = float(len(location_type_counts))
                   features[3, 6] = float(location_type_counts.iloc[0]) if len(location_type_counts) > 0 else 0.0
                   features[3, 7] = float(location_types.nunique())
               else:
                   features[3, 5] = features[3, 6] = features[3, 7] = 0.0
           else:
               features[3, 5] = features[3, 6] = features[3, 7] = 0.0


   # FIXED: Handle any remaining NaN or inf values
   features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


   return features




def build_smartwatch_edges(num_sensors):
   """
   Build edge connections for smartwatch virtual sensor graph
   """
   edges = []


   # Fully connected graph - all sensors can influence each other
   for i in range(num_sensors):
       for j in range(num_sensors):
           edges.append((i, j))


   return torch.tensor(edges, dtype=torch.long).t().contiguous()




def normalize_features(windowed_samples, motion_sensors, location_sensors, num_sensors):
    """
    Normalize all node features across windowed samples
    """
    all_features = []

    for sample in windowed_samples:
        try:
            x = build_smartwatch_features(sample['sensor_readings'], motion_sensors, location_sensors, num_sensors)
            all_features.append(x.numpy())
        except:
            continue

    all_features = np.stack(all_features)  # shape [num_samples, num_sensors, num_features_per_sensor]
    all_features_flat = all_features.reshape(-1, all_features.shape[-1])  # [total_nodes, feature_dim]

    scaler = StandardScaler()
    scaler.fit(all_features_flat)

    return scaler



def create_smartwatch_graphs(windowed_samples, sensor_to_index, activity_to_index,
                            motion_sensors, location_sensors, num_sensors, scaler=None):
   """
   Create PyG Data objects from smartwatch windowed samples (FIXED: better validation)
   """
   print("Creating smartwatch graph dataset...")


   graph_data_list = []
   failed_samples = 0


   for i, sample in enumerate(windowed_samples):
       try:
           # FIXED: Validate sample has required data
           if sample['activity'] not in activity_to_index:
               failed_samples += 1
               continue


           # Build features for virtual sensor nodes
           x = build_smartwatch_features(sample['sensor_readings'], motion_sensors,
                                         location_sensors, num_sensors)
           
           if scaler is not None:
                x = torch.tensor(scaler.transform(x), dtype=torch.float)


           # FIXED: Validate features are valid
           if torch.isnan(x).any() or torch.isinf(x).any():
               print(f"  Warning: Invalid features in sample {i}, skipping...")
               failed_samples += 1
               continue


           # Build edges (fully connected virtual sensor graph)
           edge_index = build_smartwatch_edges(num_sensors)


           # FIXED: Validate activity label
           activity_label = sample['activity']
           if activity_label not in activity_to_index:
               print(f"  Warning: Unknown activity '{activity_label}' in sample {i}, skipping...")
               failed_samples += 1
               continue


           # Activity label
           y = torch.tensor([activity_to_index[activity_label]], dtype=torch.long)


           # FIXED: Validate tensor dimensions
           if x.size(0) != num_sensors or edge_index.size(0) != 2:
               print(f"  Warning: Invalid tensor dimensions in sample {i}, skipping...")
               failed_samples += 1
               continue


           # Create graph data
           data = Data(x=x, edge_index=edge_index, y=y)
           data.participant = sample['participant']
           data.activity = sample['activity']
           data.start_time = sample['start_time']
           data.num_nodes = num_sensors


           graph_data_list.append(data)


       except Exception as e:
           print(f"Error creating smartwatch graph for sample {i}: {e}")
           failed_samples += 1
           continue


   print(f"✅ Created {len(graph_data_list)} smartwatch graph samples")
   if failed_samples > 0:
       print(f"⚠️  Failed to create {failed_samples} samples")


   # FIXED: Print final distribution
   if graph_data_list:
       activity_dist = Counter([data.activity for data in graph_data_list])
       print(f"📊 Final graph distribution: {dict(activity_dist)}")


   return graph_data_list




# =============================================================================
# STEP 2: TASK 1 - ACTIVITY RECOGNITION MODEL (SAME AS BEFORE)
# =============================================================================


class ActivityRecognitionGCN(nn.Module):
   """
   Graph Convolutional Network for Smartwatch Activity Recognition
   """


   def __init__(self, num_features, num_classes, hidden_dim=64):
       super(ActivityRecognitionGCN, self).__init__()


       self.conv1 = GCNConv(num_features, hidden_dim)
       self.conv2 = GCNConv(hidden_dim, hidden_dim)
       self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)


       self.classifier = nn.Sequential(
           nn.Linear(hidden_dim // 2, hidden_dim // 4),
           nn.ReLU(),
           nn.Dropout(0.3),
           nn.Linear(hidden_dim // 4, num_classes)
       )


       self.dropout = nn.Dropout(0.2)


   def forward(self, data):
       x, edge_index, batch = data.x, data.edge_index, data.batch


       # Graph convolutions
       x = F.relu(self.conv1(x, edge_index))
       x = self.dropout(x)
       x = F.relu(self.conv2(x, edge_index))
       x = self.dropout(x)
       x = F.relu(self.conv3(x, edge_index))


       # Global pooling
       x = global_mean_pool(x, batch)


       # Classification
       x = self.classifier(x)
       return F.log_softmax(x, dim=-1)


class ImprovedActivityGCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super(ImprovedActivityGCN, self).__init__()

        # GATConv layers (attention-based)
        self.gat1 = GATConv(num_features, hidden_dim, heads=2, concat=False)
        self.bn1 = nn.LayerNorm(hidden_dim)
        

        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.bn2 = nn.LayerNorm(hidden_dim)

        self.gat3 = GATConv(hidden_dim, hidden_dim // 2, heads=2, concat=False)
        self.bn3 = nn.LayerNorm(hidden_dim // 2)

        # Set2Set pooling for dynamic global representation
        self.set2set = Set2Set(hidden_dim // 2, processing_steps=3)

        self.res_proj = nn.Linear(hidden_dim, hidden_dim // 2)


        # Classifier with additional depth
        self.classifier = nn.Sequential(
            nn.Linear(2 * (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1
        x1 = F.relu(self.bn1(self.gat1(x, edge_index)))
        x1 = self.dropout(x1)

        # Layer 2 with residual
        x2 = F.relu(self.bn2(self.gat2(x1, edge_index)) + x1)
        x2 = self.dropout(x2)

        # Layer 3 with residual
        residual = self.res_proj(x2)
        x3 = F.relu(self.bn3(self.gat3(x2, edge_index)) + residual)

        # Global pooling (Set2Set)
        pooled = self.set2set(x3, batch)  # Output: [batch_size, 2 * hidden_dim//2]

        # Classification
        out = self.classifier(pooled)
        return F.log_softmax(out, dim=-1)

def train_activity_model(model, train_loader, val_loader, num_epochs=100):
   """
   Train smartwatch activity recognition model (FIXED: better error handling)
   """
   print("\n🎯 Training Smartwatch Activity Recognition Model...")


   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
   criterion = nn.CrossEntropyLoss()
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


   best_val_acc = 0
   patience = 0
   max_patience = 20


   train_losses, val_losses = [], []
   train_accs, val_accs = [], []


   best_model_state = None


   for epoch in range(num_epochs):
       # Training
       model.train()
       total_loss = 0
       correct = 0
       total = 0


       for batch_idx, batch in enumerate(train_loader):
           try:
               batch = batch.to(DEVICE)


               # FIXED: Validate batch before processing
               if batch.y.size(0) == 0:
                   print(f"Warning: Empty batch {batch_idx} in training, skipping...")
                   continue


               optimizer.zero_grad()


               out = model(batch)


               # FIXED: Ensure output and target dimensions match
               target = batch.y.view(-1)
               if out.size(0) != target.size(0):
                   print(f"Warning: Dimension mismatch in batch {batch_idx}, skipping...")
                   continue


               loss = criterion(out, target)


               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               optimizer.step()


               total_loss += loss.item()
               pred = out.argmax(dim=1)
               correct += (pred == target).sum().item()
               total += target.size(0)


           except Exception as e:
               print(f"Error in training batch {batch_idx}: {e}")
               continue


       if total == 0:
           print("Warning: No valid training batches processed!")
           continue


       train_loss = total_loss / len(train_loader)
       train_acc = correct / total


       # Validation
       model.eval()
       val_loss = 0
       val_correct = 0
       val_total = 0


       with torch.no_grad():
           for batch_idx, batch in enumerate(val_loader):
               try:
                   batch = batch.to(DEVICE)


                   # FIXED: Validate batch before processing
                   if batch.y.size(0) == 0:
                       continue


                   out = model(batch)
                   target = batch.y.squeeze()


                   if out.size(0) != target.size(0):
                       continue


                   loss = criterion(out, target)


                   val_loss += loss.item()
                   pred = out.argmax(dim=1)
                   val_correct += (pred == target).sum().item()
                   val_total += target.size(0)


               except Exception as e:
                   print(f"Error in validation batch {batch_idx}: {e}")
                   continue


       if val_total == 0:
           print("Warning: No valid validation batches processed!")
           continue


       val_loss /= len(val_loader)
       val_acc = val_correct / val_total


       # Learning rate scheduling
       scheduler.step(val_loss)


       # Early stopping
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           patience = 0
           best_model_state = model.state_dict().copy()
           torch.save(model.state_dict(), 'best_smartwatch_activity_model.pth')
       else:
           patience += 1


       # Logging
       if epoch % 10 == 0:
           print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                 f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


       train_losses.append(train_loss)
       val_losses.append(val_loss)
       train_accs.append(train_acc)
       val_accs.append(val_acc)


       if patience >= max_patience:
           print(f"Early stopping at epoch {epoch}")
           break


   # Load best model
   if best_model_state is not None:
       model.load_state_dict(best_model_state)
   elif os.path.exists('best_smartwatch_activity_model.pth'):
       model.load_state_dict(torch.load('best_smartwatch_activity_model.pth'))


   print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")


   return train_losses, val_losses, train_accs, val_accs




def evaluate_activity_model(model, test_loader, activity_to_index):
   """
   Evaluate smartwatch activity recognition model
   """
   print("\n📊 Evaluating Smartwatch Activity Recognition Model...")


   model.eval()
   all_preds = []
   all_labels = []
   all_probs = []


   with torch.no_grad():
       for batch in test_loader:
           try:
               batch = batch.to(DEVICE)
               if batch.y.size(0) == 0:
                   continue
               out = model(batch)
               pred = out.argmax(dim=1)
               probs = F.softmax(out, dim=1)
               all_preds.extend(pred.cpu().numpy())
               all_labels.extend(batch.y.view(-1).cpu().numpy())
               all_probs.extend(probs.cpu().numpy())
           except Exception as e:
               print(f"Error in evaluation batch: {e}")
               continue


   if len(all_preds) == 0:
       print("Error: No valid predictions made!")
       return 0.0, 0.0, [], [], []


   accuracy = accuracy_score(all_labels, all_preds)
   f1 = f1_score(all_labels, all_preds, average='weighted')


   activity_names = [name for name, _ in sorted(activity_to_index.items(), key=lambda x: x[1])]
   all_class_indices = list(range(len(activity_names)))
   report = classification_report(
       all_labels, all_preds,
       target_names=activity_names,
       labels=all_class_indices,
       zero_division=0  # 防止没有样本时报警告
   )


   print(f"Test Accuracy: {accuracy:.4f}")
   print(f"Test F1-Score: {f1:.4f}")
   print("\nDetailed Classification Report:")
   print(report)


   return accuracy, f1, all_preds, all_labels, all_probs




# =============================================================================
# STEP 3: TASK 2 - ANOMALY DETECTION MODEL (SAME AS BEFORE)
# =============================================================================


class GraphAutoEncoder(nn.Module):
   """
   Graph AutoEncoder for Smartwatch Anomaly Detection
   """


   def __init__(self, num_features, hidden_dim=32):
       super(GraphAutoEncoder, self).__init__()


       # Encoder
       self.encoder1 = GCNConv(num_features, hidden_dim * 2)
       self.encoder2 = GCNConv(hidden_dim * 2, hidden_dim)


       # Decoder (reconstruction)
       self.decoder1 = GCNConv(hidden_dim, hidden_dim * 2)
       self.decoder2 = GCNConv(hidden_dim * 2, num_features)


       self.dropout = nn.Dropout(0.1)


   def encode(self, x, edge_index):
       h1 = F.relu(self.encoder1(x, edge_index))
       h1 = self.dropout(h1)
       h2 = F.relu(self.encoder2(h1, edge_index))
       return h2


   def decode(self, z, edge_index):
       h1 = F.relu(self.decoder1(z, edge_index))
       h1 = self.dropout(h1)
       h2 = self.decoder2(h1, edge_index)
       return h2


   def forward(self, data):
       x, edge_index = data.x, data.edge_index
       z = self.encode(x, edge_index)
       x_recon = self.decode(z, edge_index)
       return x_recon, z

class ImprovedGraphAutoEncoder(nn.Module):
    """
    Improved Graph AutoEncoder for Anomaly Detection
    """
    def __init__(self, num_features, hidden_dim=64):
        super(ImprovedGraphAutoEncoder, self).__init__()

        # Encoder
        self.encoder1 = GCNConv(num_features, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

        # Bottleneck LSTM (optional temporal modeling for node embeddings)
        self.use_lstm = False
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Decoder
        self.decoder1 = GCNConv(hidden_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, num_features)
        self.norm3 = LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def encode(self, x, edge_index):
        h = F.leaky_relu(self.encoder1(x, edge_index))
        h = self.norm1(h)
        h = self.dropout(h)

        h2 = F.leaky_relu(self.encoder2(h, edge_index))
        h2 = self.norm2(h2)

        # Optional residual
        z = h2 + h  # residual connection
        return z

    def decode(self, z, edge_index):
        h = F.leaky_relu(self.decoder1(z, edge_index))
        h = self.norm3(h)
        h = self.dropout(h)

        out = self.decoder2(h, edge_index)
        return out

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encode(x, edge_index)

        if self.use_lstm:
            # Assume input is a single graph (not a sequence), so model temporal dynamics over nodes
            z_seq = z.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            lstm_out, _ = self.lstm(z_seq)
            z = lstm_out.squeeze(1)  # [num_nodes, hidden_dim]

        x_recon = self.decode(z, edge_index)
        return x_recon, z


def train_anomaly_model(model, train_loader, val_loader, num_epochs=100):
   """
   Train smartwatch anomaly detection model (FIXED: same error handling as activity model)
   """
   print("\n🚨 Training Smartwatch Anomaly Detection Model...")


   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
   criterion = nn.MSELoss()
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


   best_val_loss = float('inf')
   patience = 0
   max_patience = 20


   train_losses, val_losses = [], []
   best_model_state = None


   for epoch in range(num_epochs):
       # Training
       model.train()
       total_loss = 0
       valid_batches = 0


       for batch_idx, batch in enumerate(train_loader):
           try:
               batch = batch.to(DEVICE)


               if batch.x.size(0) == 0:
                   continue


               optimizer.zero_grad()


               x_recon, z = model(batch)
               loss = criterion(x_recon, batch.x)
               
            #    print(f"Batch {batch_idx} | x shape: {batch.x.shape} | x_recon shape: {x_recon.shape}") 
            #    print(f"Batch {batch_idx} | x min: {batch.x.min().item():.4f}, max: {batch.x.max().item():.4f}, mean: {batch.x.mean().item():.4f}")
            #    print(f"Batch {batch_idx} | x_recon min: {x_recon.min().item():.4f}, max: {x_recon.max().item():.4f}, mean: {x_recon.mean().item():.4f}")
            #    print(f"Batch {batch_idx} | MSE Loss: {loss.item():.4f}")


               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               optimizer.step()


               total_loss += loss.item()
               valid_batches += 1


           except Exception as e:
               print(f"Error in anomaly training batch {batch_idx}: {e}")
               continue


       if valid_batches == 0:
           print("Warning: No valid training batches for anomaly model!")
           continue


       train_loss = total_loss / valid_batches


       # Validation
       model.eval()
       val_loss = 0
       valid_val_batches = 0


       with torch.no_grad():
           for batch_idx, batch in enumerate(val_loader):
               try:
                   batch = batch.to(DEVICE)


                   if batch.x.size(0) == 0:
                       continue


                   x_recon, z = model(batch)
                   loss = criterion(x_recon, batch.x)
                   val_loss += loss.item()
                   valid_val_batches += 1


               except Exception as e:
                   print(f"Error in anomaly validation batch {batch_idx}: {e}")
                   continue


       if valid_val_batches == 0:
           print("Warning: No valid validation batches for anomaly model!")
           continue


       val_loss /= valid_val_batches


       # Learning rate scheduling
       scheduler.step(val_loss)


       # Early stopping
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience = 0
           best_model_state = model.state_dict().copy()
           torch.save(model.state_dict(), 'best_smartwatch_anomaly_model.pth')
       else:
           patience += 1


           # Logging
       if epoch % 10 == 0:
           print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")


       train_losses.append(train_loss)
       val_losses.append(val_loss)


       if patience >= max_patience:
           print(f"Early stopping at epoch {epoch}")
           break


       # Load best model
   if best_model_state is not None:
       model.load_state_dict(best_model_state)
   elif os.path.exists('best_smartwatch_anomaly_model.pth'):
       model.load_state_dict(torch.load('best_smartwatch_anomaly_model.pth'))


   print(f"✅ Training completed. Best validation loss: {best_val_loss:.6f}")


   return train_losses, val_losses




def detect_anomalies(model, test_loader, threshold_percentile=95):
   """
   Detect anomalies in smartwatch data using reconstruction error
   """
   print("\n🔍 Detecting Smartwatch Anomalies...")


   model.eval()
   reconstruction_errors = []
   graph_info = []


   with torch.no_grad():
       for batch in test_loader:
           try:
               batch = batch.to(DEVICE)


               if batch.x.size(0) == 0:
                   continue


               x_recon, z = model(batch)


               # Calculate reconstruction error for each graph
               batch_size = batch.num_graphs
               batch_errors = []


               for i in range(batch_size):
                   # Get nodes for this graph
                   if batch.batch is not None:
                       mask = batch.batch == i
                       x_orig = batch.x[mask]
                       x_rec = x_recon[mask]
                   else:
                       x_orig = batch.x
                       x_rec = x_recon


                   # Calculate MSE for this graph
                   error = F.mse_loss(x_rec, x_orig, reduction='mean')
                   reconstruction_errors.append(error.item())
                   batch_errors.append(error.item())


               # Store graph information
               for i in range(batch_size):
                   graph_info.append({
                       'participant': getattr(batch, 'participant', ['unknown'] * batch_size)[i] if hasattr(batch,
                                                                                                            'participant') else 'unknown',
                       'activity': getattr(batch, 'activity', ['unknown'] * batch_size)[i] if hasattr(batch,
                                                                                                      'activity') else 'unknown',
                       'error': batch_errors[i]
                   })


           except Exception as e:
               print(f"Error in anomaly detection: {e}")
               continue


   if len(reconstruction_errors) == 0:
       print("Error: No reconstruction errors calculated!")
       return [], [], 0.0, []


   # Set threshold
   threshold = np.percentile(reconstruction_errors, threshold_percentile)
   anomalies = [error > threshold for error in reconstruction_errors]


   # Analyze results
   anomaly_count = sum(anomalies)
   total_count = len(reconstruction_errors)
   anomaly_rate = anomaly_count / total_count * 100


   print(f"Reconstruction error threshold (P{threshold_percentile}): {threshold:.6f}")
   print(f"Anomalies detected: {anomaly_count}/{total_count} ({anomaly_rate:.1f}%)")


   # Analyze anomalies by activity and participant
   anomaly_info = [info for i, info in enumerate(graph_info) if anomalies[i]]


   if anomaly_info:
       print("\n🚨 Smartwatch Anomaly Analysis:")
       activity_anomalies = Counter([info['activity'] for info in anomaly_info])
       participant_anomalies = Counter([info['participant'] for info in anomaly_info])


       print("By Activity:")
       for activity, count in activity_anomalies.most_common():
           print(f"  {activity}: {count} anomalies")


       print("By Participant:")
       for participant, count in participant_anomalies.most_common(5):
           print(f"  {participant}: {count} anomalies")


   return reconstruction_errors, anomalies, threshold, graph_info




# =============================================================================
# STEP 4: MAIN EXECUTION PIPELINE FOR SW DATASET (FIXED)
# =============================================================================


def main():
    """
    Complete SW smartwatch pipeline execution (FIXED: better error handling and validation)
    """
    print("🚀 Starting Complete SW Smartwatch Pipeline...")
    start_time = time.time()
    results_table = []

    try:
        df_all = load_sw_dataset()

        if len(df_all) < 100:
            raise ValueError(f"Insufficient data: only {len(df_all)} samples found")

        sensor_to_index, activity_to_index, motion_sensors, location_sensors, num_sensors, num_activities = build_smartwatch_mappings(df_all)

        if num_activities < 2:
            raise ValueError("Need at least 2 activities for classification.")

        for window_size in [30, 35, 40, 45, 50, 60]:
            for stride in [1, 3, 5, 10, 15]:
                if window_size <= stride:
                    print(f"Skipping invalid configuration: window_size={window_size}, stride={stride}")
                    continue
                if window_size == 80:
                    if stride != 5 and stride != 10 and stride != 25:
                        continue
                if window_size == 100:
                    if stride != 5:
                        continue


                print(f"\n🔁 Testing: window_size={window_size}, stride={stride}")
                windowed_samples = create_smartwatch_windows(df_all, window_size, stride)

                # FIXED: Check if we have enough windows
                if len(windowed_samples) < 10:
                    raise ValueError(f"Insufficient windowed samples: only {len(windowed_samples)} created")


                sensor_to_index, activity_to_index, motion_sensors, location_sensors, num_sensors, num_activities = build_smartwatch_mappings(
                    df_all)


                # FIXED: Validate we have activities to classify
                if num_activities < 2:
                    raise ValueError(f"Need at least 2 activities for classification, found {num_activities}")


                # Step 2: Create smartwatch graph dataset
                scaler = normalize_features(windowed_samples, motion_sensors, location_sensors, num_sensors)

                graph_data_list = create_smartwatch_graphs(windowed_samples, sensor_to_index, activity_to_index,
                                                            motion_sensors, location_sensors, num_sensors, scaler=scaler)


                # FIXED: More stringent check for graph samples
                if len(graph_data_list) < 20:
                    raise ValueError(f"Insufficient graph samples for training: only {len(graph_data_list)} created")


                # FIXED: Validate all graphs have proper labels
                valid_graphs = []
                for i, data in enumerate(graph_data_list):
                    if data.y.size(0) > 0 and not torch.isnan(data.y).any() and not torch.isnan(data.x).any():
                        valid_graphs.append(data)
                    else:
                        print(f"Removing invalid graph {i}")


                graph_data_list = valid_graphs


                if len(graph_data_list) < 20:
                    raise ValueError(f"After validation, insufficient graphs: only {len(graph_data_list)} remaining")


                # Step 3: Split data by participants (prevent data leakage)
                participants = list(set([data.participant for data in graph_data_list]))


                # FIXED: Ensure we have enough participants for meaningful splits
                if len(participants) < 3:
                    print(f"Warning: Only {len(participants)} participants found, using simple random split")
                    train_data, temp_data = train_test_split(graph_data_list, test_size=0.4, random_state=42)
                    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
                    participant_train = participant_val = participant_test = participants
                else:
                    participant_train, participant_temp = train_test_split(participants, test_size=0.4, random_state=42)
                    participant_val, participant_test = train_test_split(participant_temp, test_size=0.5, random_state=42)


                    train_data = [data for data in graph_data_list if data.participant in participant_train]
                    val_data = [data for data in graph_data_list if data.participant in participant_val]
                    test_data = [data for data in graph_data_list if data.participant in participant_test]


                # FIXED: Ensure all splits have sufficient data
                min_samples = 5
                if len(train_data) < min_samples or len(val_data) < min_samples or len(test_data) < min_samples:
                    print(
                        f"Warning: Small splits detected. Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
                    # Fallback to simple random split
                    train_data, temp_data = train_test_split(graph_data_list, test_size=0.4, random_state=42)
                    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


                print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
                print(
                    f"👥 Participant split: Train={len(participant_train)}, Val={len(participant_val)}, Test={len(participant_test)}")


                # Step 4: Create data loaders
                batch_size = min(8, max(1, len(train_data) // 10))  # FIXED: More conservative batch size
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


                num_features = graph_data_list[0].x.shape[1]
                print(f"🔧 Model config: {num_features} features, {num_activities} classes, {num_sensors} virtual sensors")
                print(f"📦 Batch size: {batch_size}")


                # Step 5: Task 1 - Smartwatch Activity Recognition
                print("\n" + "=" * 60)
                print("TASK 1: SMARTWATCH ACTIVITY RECOGNITION")
                print("=" * 60)


                activity_model = ActivityRecognitionGCN(num_features, num_activities).to(DEVICE)
                train_losses, val_losses, train_accs, val_accs = train_activity_model(
                    activity_model, train_loader, val_loader, num_epochs=100  
                )


                activity_accuracy, activity_f1, preds, labels, probs = evaluate_activity_model(
                    activity_model, test_loader, activity_to_index
                )

                improved_activity_model = ImprovedActivityGCN(num_features, num_activities).to(DEVICE)
                improved_train_losses, improved_val_losses, improved_train_accs, improved_val_accs = train_activity_model(
                    improved_activity_model, train_loader, val_loader, num_epochs=100  
                )


                improved_activity_accuracy, improved_activity_f1, improved_preds, improved_labels, improved_probs = evaluate_activity_model(
                    improved_activity_model, test_loader, activity_to_index
                )
                


                # Step 6: Task 2 - Smartwatch Anomaly Detection
                print("\n" + "=" * 60)
                print("TASK 2: SMARTWATCH ANOMALY DETECTION")
                print("=" * 60)


                anomaly_model = GraphAutoEncoder(num_features, hidden_dim=16).to(DEVICE)  # FIXED: Smaller hidden dim
                anomaly_train_losses, anomaly_val_losses = train_anomaly_model(
                    anomaly_model, train_loader, val_loader, num_epochs=100 
                )


                reconstruction_errors, anomalies, threshold, anomaly_graph_info = detect_anomalies(
                    anomaly_model, test_loader, threshold_percentile=95
                )

                improved_anomaly_model = ImprovedGraphAutoEncoder(num_features, hidden_dim=16).to(DEVICE)  # FIXED: Smaller hidden dim
                improved_anomaly_train_losses, improved_anomaly_val_losses = train_anomaly_model(
                    improved_anomaly_model, train_loader, val_loader, num_epochs=100  
                )


                improved_reconstruction_errors, improved_anomalies, improved_threshold, improved_anomaly_graph_info = detect_anomalies(
                    improved_anomaly_model, test_loader, threshold_percentile=95
                )
                

                # Step 7: Final Results Summary for Smartwatch Dataset
                total_time = (time.time() - start_time) / 60

                print("\n" + "=" * 60)
                print("🎉 SMARTWATCH PIPELINE RESULTS SUMMARY")
                print("=" * 60)
                print(f"⏱️  Total execution time: {total_time:.2f} minutes")
                print(f"📊 Total graphs: {len(graph_data_list)} from {len(participants)} participants")
                print(f"📱 Smartwatch features: {num_features} per node (motion, location, battery, context)")
                print(f"🔌 Graph structure: Virtual sensors as nodes with temporal edges")

                # === Activity Recognition Results ===
                print("\n🎯 TASK 1 - ACTIVITY RECOGNITION")
                print(f"   • Baseline Accuracy: {activity_accuracy:.4f}")
                print(f"   • Baseline F1-Score: {activity_f1:.4f}")
                print(f"   • Improved Accuracy: {improved_activity_accuracy:.4f}")
                print(f"   • Improved Accuracy: {improved_activity_f1:.4f}")
                print(f"   • Activities Detected: {list(activity_to_index.keys())}")

                results_table.append({
                    'window_size': window_size,
                    'stride': stride,
                    'num_graphs': len(graph_data_list),
                    'activity_acc': activity_accuracy,
                    'activity_f1': activity_f1,
                    'improved_activity_acc': improved_activity_accuracy,
                    'improved_activity_f1': improved_activity_f1,
                    'anomaly_rate': sum(anomalies)/len(anomalies)*100 if anomalies else 0,
                    'improved_anomaly_rate': sum(improved_anomalies)/len(improved_anomalies)*100 if improved_anomalies else 0,
                })

                if len(preds) > 0 and len(labels) > 0 and len(improved_preds) > 0 and len(improved_labels) > 0:
                    baseline_performance = {}
                    improved_performance = {}
                    for activity, idx in activity_to_index.items():
                        # Baseline model
                        mask_baseline = np.array(labels) == idx
                        if mask_baseline.sum() > 0:
                            acc = (np.array(preds)[mask_baseline] == idx).mean()
                            baseline_performance[activity] = acc

                        # Improved model
                        mask_improved = np.array(improved_labels) == idx
                        if mask_improved.sum() > 0:
                            acc_improved = (np.array(improved_preds)[mask_improved] == idx).mean()
                            improved_performance[activity] = acc_improved

                    print("\n📈 Activity-wise Accuracy:")
                    for activity in sorted(activity_to_index.keys()):
                        base = baseline_performance.get(activity, 0.0)
                        imp = improved_performance.get(activity, 0.0)
                        print(f"   • {activity}: Baseline = {base:.2f}, Improved = {imp:.2f}")

                # === Anomaly Detection Results ===
                print("\n🚨 TASK 2 - ANOMALY DETECTION")
                print(f"   • Baseline Anomaly Rate: {sum(anomalies) / len(anomalies) * 100:.2f}%" if anomalies else "   • No anomalies detected")
                print(f"   • Baseline Threshold (P95): {threshold:.6f}")
                print(f"   • Baseline Total Anomalies: {sum(anomalies)}/{len(anomalies)}" if anomalies else "   • Total Anomalies: 0/0")

                print(f"   • Improved Anomaly Rate: {sum(improved_reconstruction_errors > threshold) / len(improved_reconstruction_errors) * 100:.2f}%")
                print(f"   • Improved Threshold (P95): {improved_threshold:.6f}")
                print(f"   • Improved Total Anomalies: {sum(improved_anomalies)}/{len(improved_anomalies)}" if improved_anomalies else "   • Total Anomalies: 0/0")

                # === Data Summary ===
                print("\n📊 DATASET INSIGHTS")
                print(f"   • Participants analyzed: {len(participants)}")
                print(f"   • Total sensor readings: {len(df_all)}")
                print(f"   • Window size: 30 timesteps with 50% overlap")
                print(f"   • Sensor fusion strategy: Aggregated motion, location, and context data")

                print("\n✅ Smartwatch pipeline completed successfully!")
        
    except Exception as e:
       print(f"❌ SW Pipeline failed: {e}")
       import traceback
       traceback.print_exc()
       return None


    results_df = pd.DataFrame(results_table)
    print("\n📊 Window/Stride Evaluation Summary:")
    print(results_df.to_string(index=False))

    # Optionally save as CSV
    results_df.to_csv("sw_results4.csv", index=False)


if __name__ == "__main__":
   # Run the complete SW smartwatch pipeline
   results = main()

