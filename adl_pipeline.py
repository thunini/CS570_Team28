"""
Complete CASAS Pipeline: Graph Construction + Activity Recognition + Anomaly Detection
Based on the proven CASAS dataset structure with real activity labels
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GAE, VGAE, global_max_pool, LayerNorm, Set2Set

from torch_geometric.utils import negative_sampling

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# =============================================================================
# STEP 1: DATA LOADING AND GRAPH CONSTRUCTION (Proven Approach)
# =============================================================================

def load_casas_dataset(data_dir='data/CASAS_smart_home/'):
    """
    Load CASAS dataset with proper activity labels
    """
    print("Loading CASAS dataset...")
    
    # Define subject IDs
    subject_ids = [f"P{str(i).zfill(2)}" for i in range(1, 17)] + ["P32", "P40", "P41", "P42", "P43", "P49", "P50", "P51"]
    
    # Map file index to activity name
    activity_map = {
        1: "Phone_Call",
        2: "Wash_hands", 
        3: "Cook",
        4: "Eat",
        5: "Clean"
    }
    
    # Load and merge data
    all_subject_data = []
    files_found = 0
    
    for subject in subject_ids:
        subject_id = subject.lower()
        for idx, activity in activity_map.items():
            filename = os.path.join(data_dir, f"{subject_id}.t{idx}")
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename, sep='\t', header=None, 
                                   names=['date', 'time', 'sensor_id', 'state'], 
                                   engine='python')
                    df['activity'] = activity
                    df['subject'] = subject
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                    df = df.dropna(subset=['timestamp'])
                    if len(df) > 0:
                        all_subject_data.append(df)
                        files_found += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    if not all_subject_data:
        raise ValueError(f"No data files found in {data_dir}")
    
    # Combine and sort data
    df_all = pd.concat(all_subject_data, ignore_index=True)
    df_all = df_all.sort_values(by=['subject', 'timestamp']).reset_index(drop=True)
    
    print(f"✅ Loaded {files_found} files, {len(df_all)} sensor events")
    print(f"📊 Subjects: {len(df_all['subject'].unique())}")
    print(f"🎯 Activities: {df_all['activity'].value_counts().to_dict()}")
    
    return df_all, activity_map

def create_sliding_windows(df_all, window_size=10, stride=5):
    """
    Create sliding windows for temporal graphs
    """
    print(f"Creating sliding windows (size={window_size}, stride={stride})...")
    
    windowed_samples = []
    grouped = df_all.groupby(['subject', 'activity'])
    
    for (subject, activity), group in grouped:
        group = group.sort_values('timestamp').reset_index(drop=True)
        max_start = len(group) - window_size + 1
        
        if max_start > 0:
            for start_idx in range(0, max_start, stride):
                end_idx = start_idx + window_size
                window_df = group.iloc[start_idx:end_idx]
                if len(window_df) == window_size:
                    windowed_samples.append({
                        'subject': subject,
                        'activity': activity,
                        'events': window_df,
                        'start_time': window_df.iloc[0]['timestamp'],
                        'end_time': window_df.iloc[-1]['timestamp']
                    })
    
    print(f"✅ Created {len(windowed_samples)} windowed samples")
    return windowed_samples

def build_sensor_mappings(df_all):
    """
    Build sensor ID mappings and infer sensor types
    """
    # Sensor ID mapping
    sensor_id_to_index = {sid: idx for idx, sid in enumerate(sorted(df_all['sensor_id'].unique()))}
    num_sensors = len(sensor_id_to_index)
    
    # Activity mapping
    activity_to_index = {activity: idx for idx, activity in enumerate(sorted(df_all['activity'].unique()))}
    num_activities = len(activity_to_index)
    
    # Infer sensor types
    sensor_types = {}
    for sid in df_all['sensor_id'].unique():
        states = df_all[df_all['sensor_id'] == sid]['state'].unique()
        if any(s in ['ON', 'OFF'] for s in states):
            sensor_types[sid] = 'M'  # Motion
        elif any(s in ['PRESENT', 'ABSENT'] for s in states):
            sensor_types[sid] = 'I'  # Item
        elif any(s in ['OPEN', 'CLOSE'] for s in states):
            sensor_types[sid] = 'D'  # Door
        elif any(str(s).replace('.', '', 1).replace('-', '', 1).isdigit() for s in states):
            sensor_types[sid] = 'AD'  # Analog
        else:
            sensor_types[sid] = 'UNKNOWN'
    
    print(f"📡 Sensors: {num_sensors} total")
    print(f"🔧 Sensor types: {Counter(sensor_types.values())}")
    print(f"🎯 Activities: {num_activities} classes")
    
    return sensor_id_to_index, activity_to_index, sensor_types, num_sensors, num_activities

def build_graph_features(events, sensor_types, sensor_id_to_index, num_sensors):
    """
    Build enhanced feature matrix for graph nodes
    """
    # Enhanced features: [count_1, count_2, analog_avg, total_activations, time_span]
    features = torch.zeros((num_sensors, 5), dtype=torch.float32)
    
    time_span = (events['timestamp'].max() - events['timestamp'].min()).total_seconds()
    
    for sid in events['sensor_id'].unique():
        idx = sensor_id_to_index[sid]
        stype = sensor_types.get(sid, 'UNKNOWN')
        sensor_events = events[events['sensor_id'] == sid]
        
        # Type-specific counts
        if stype == 'M':
            count_1 = (sensor_events['state'] == 'ON').sum()
            count_2 = (sensor_events['state'] == 'OFF').sum()
        elif stype == 'I':
            count_1 = (sensor_events['state'] == 'PRESENT').sum()
            count_2 = (sensor_events['state'] == 'ABSENT').sum()
        elif stype == 'D':
            count_1 = (sensor_events['state'] == 'OPEN').sum()
            count_2 = (sensor_events['state'] == 'CLOSE').sum()
        else:
            count_1 = count_2 = 0
        
        # Analog average
        analog_values = pd.to_numeric(sensor_events['state'], errors='coerce').dropna()
        analog_avg = float(analog_values.mean()) if not analog_values.empty else 0.0
        
        # Additional features
        total_activations = len(sensor_events)
        time_span_norm = time_span / 3600.0  # Hours
        
        features[idx] = torch.tensor([count_1, count_2, analog_avg, total_activations, time_span_norm])
    
    return features

def build_temporal_edges(events, sensor_id_to_index):
    """
    Build temporal edges based on sensor activation sequence
    """
    sorted_events = events.sort_values("timestamp").reset_index(drop=True)
    sensor_sequence = sorted_events['sensor_id'].tolist()
    
    edges = []
    # Sequential edges
    for i in range(len(sensor_sequence) - 1):
        src = sensor_id_to_index[sensor_sequence[i]]
        tgt = sensor_id_to_index[sensor_sequence[i + 1]]
        edges.append((src, tgt))
    
    # Add self-loops for active sensors
    for sid in events['sensor_id'].unique():
        idx = sensor_id_to_index[sid]
        edges.append((idx, idx))
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

def create_graph_dataset(windowed_samples, sensor_id_to_index, activity_to_index, sensor_types, num_sensors):
    """
    Create PyG Data objects from windowed samples
    """
    print("Creating graph dataset...")
    
    graph_data_list = []
    
    for sample in windowed_samples:
        try:
            # Build features
            x = build_graph_features(sample['events'], sensor_types, sensor_id_to_index, num_sensors)
            
            # Build edges
            edge_index = build_temporal_edges(sample['events'], sensor_id_to_index)
            
            # Label
            y = torch.tensor([activity_to_index[sample['activity']]], dtype=torch.long)
            
            # Create graph data
            data = Data(x=x, edge_index=edge_index, y=y)
            data.subject = sample['subject']
            data.activity = sample['activity']
            data.start_time = sample['start_time']
            data.num_nodes = num_sensors
            
            graph_data_list.append(data)
            
        except Exception as e:
            print(f"Error creating graph: {e}")
            continue
    
    print(f"✅ Created {len(graph_data_list)} graph samples")
    return graph_data_list

# =============================================================================
# STEP 2: TASK 1 - ACTIVITY RECOGNITION MODEL
# =============================================================================

class ActivityRecognitionGCN(nn.Module):
    """
    Graph Convolutional Network for Activity Recognition
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
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gat3 = GATConv(hidden_dim, hidden_dim // 2, heads=2, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # Set2Set pooling for dynamic global representation
        self.set2set = Set2Set(hidden_dim // 2, processing_steps=3)

        self.res_proj = nn.Linear(hidden_dim, hidden_dim // 2)


        # Classifier with additional depth
        self.classifier = nn.Sequential(
            nn.Linear(2 * (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
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
    Train activity recognition model
    """
    print("\n🎯 Training Activity Recognition Model...")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    patience = 0
    max_patience = 20
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.squeeze()).sum().item()
            total += batch.y.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                loss = criterion(out, batch.y.squeeze())
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y.squeeze()).sum().item()
                val_total += batch.y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'best_activity_model.pth')
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
    model.load_state_dict(torch.load('best_activity_model.pth'))
    print(f"✅ Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_activity_model(model, test_loader, activity_to_index):
    """
    Evaluate activity recognition model
    """
    print("\n📊 Evaluating Activity Recognition Model...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            
            pred = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Detailed classification report
    activity_names = [name for name, _ in sorted(activity_to_index.items(), key=lambda x: x[1])]
    report = classification_report(all_labels, all_preds, target_names=activity_names)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    return accuracy, f1, all_preds, all_labels, all_probs

# =============================================================================
# STEP 3: TASK 2 - ANOMALY DETECTION MODEL
# =============================================================================

class GraphAutoEncoder(nn.Module):
    """
    Graph AutoEncoder for Anomaly Detection
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
    Train anomaly detection model
    """
    print("\n🚨 Training Anomaly Detection Model...")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 0
    max_patience = 20
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            x_recon, z = model(batch)
            loss = criterion(x_recon, batch.x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                x_recon, z = model(batch)
                loss = criterion(x_recon, batch.x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_anomaly_model.pth')
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
    model.load_state_dict(torch.load('best_anomaly_model.pth'))
    print(f"✅ Training completed. Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses

def detect_anomalies(model, test_loader, threshold_percentile=95):
    """
    Detect anomalies using reconstruction error
    """
    print("\n🔍 Detecting Anomalies...")
    
    model.eval()
    reconstruction_errors = []
    graph_info = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
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
                    'subject': getattr(batch, 'subject', ['unknown'] * batch_size)[i] if hasattr(batch, 'subject') else 'unknown',
                    'activity': getattr(batch, 'activity', ['unknown'] * batch_size)[i] if hasattr(batch, 'activity') else 'unknown',
                    'error': batch_errors[i]
                })
    
    # Set threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    anomalies = [error > threshold for error in reconstruction_errors]
    
    # Analyze results
    anomaly_count = sum(anomalies)
    total_count = len(reconstruction_errors)
    anomaly_rate = anomaly_count / total_count * 100
    
    print(f"Reconstruction error threshold (P{threshold_percentile}): {threshold:.6f}")
    print(f"Anomalies detected: {anomaly_count}/{total_count} ({anomaly_rate:.1f}%)")
    
    # Analyze anomalies by activity and subject
    anomaly_info = [info for i, info in enumerate(graph_info) if anomalies[i]]
    
    if anomaly_info:
        print("\n🚨 Anomaly Analysis:")
        activity_anomalies = Counter([info['activity'] for info in anomaly_info])
        subject_anomalies = Counter([info['subject'] for info in anomaly_info])
        
        print("By Activity:")
        for activity, count in activity_anomalies.most_common():
            print(f"  {activity}: {count} anomalies")
        
        print("By Subject:")
        for subject, count in subject_anomalies.most_common(5):
            print(f"  {subject}: {count} anomalies")
    
    return reconstruction_errors, anomalies, threshold, graph_info

# =============================================================================
# STEP 4: MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Complete pipeline execution
    """
    print("🚀 Starting Complete CASAS Pipeline...")
    start_time = time.time()
    results_table = []
    
    try:
        # Step 1: Load and preprocess data
        df_all, activity_map = load_casas_dataset()

        sensor_id_to_index, activity_to_index, sensor_types, num_sensors, num_activities = build_sensor_mappings(df_all)

        
        for window_size in [10, 15, 20, 25, 30]:
            for stride in [0, 1, 5, 10]:
                print(f"\n🔁 Testing: window_size={window_size}, stride={stride}")


                windowed_samples = create_sliding_windows(df_all, window_size=10, stride=5)        
                # Step 2: Create graph dataset
                graph_data_list = create_graph_dataset(windowed_samples, sensor_id_to_index, activity_to_index, sensor_types, num_sensors)
                
                if len(graph_data_list) < 10:
                    raise ValueError("Not enough graph samples for training")
                
                # Step 3: Split data by subjects (prevent data leakage)
                subjects = list(set([data.subject for data in graph_data_list]))
                subject_train, subject_temp = train_test_split(subjects, test_size=0.4, random_state=42)
                subject_val, subject_test = train_test_split(subject_temp, test_size=0.5, random_state=42)
                
                train_data = [data for data in graph_data_list if data.subject in subject_train]
                val_data = [data for data in graph_data_list if data.subject in subject_val]
                test_data = [data for data in graph_data_list if data.subject in subject_test]
                
                print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
                print(f"👥 Subject split: Train={len(subject_train)}, Val={len(subject_val)}, Test={len(subject_test)}")
                
                # Step 4: Create data loaders
                batch_size = min(32, len(train_data) // 4)
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                
                num_features = graph_data_list[0].x.shape[1]
                print(f"🔧 Model config: {num_features} features, {num_activities} classes")
                
                # Step 5: Task 1 - Activity Recognition
                print("\n" + "="*60)
                print("TASK 1: ACTIVITY RECOGNITION")
                print("="*60)
                
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
                
                # Step 6: Task 2 - Anomaly Detection
                print("\n" + "="*60)
                print("TASK 2: ANOMALY DETECTION")
                print("="*60)
                
                anomaly_model = GraphAutoEncoder(num_features, hidden_dim=32).to(DEVICE)
                anomaly_train_losses, anomaly_val_losses = train_anomaly_model(
                    anomaly_model, train_loader, val_loader, num_epochs=100
                )
                
                reconstruction_errors, anomalies, threshold, anomaly_graph_info = detect_anomalies(
                    anomaly_model, test_loader, threshold_percentile=95
                )

                improved_anomaly_model = ImprovedGraphAutoEncoder(num_features, hidden_dim=32).to(DEVICE)
                improved_anomaly_train_losses, improved_anomaly_val_losses = train_anomaly_model(
                    improved_anomaly_model, train_loader, val_loader, num_epochs=100  
                )


                improved_reconstruction_errors, improved_anomalies, improved_threshold, improved_anomaly_graph_info = detect_anomalies(
                    improved_anomaly_model, test_loader, threshold_percentile=95
                )
                
                # Step 7: Final Results Summary
                total_time = (time.time() - start_time) / 60
                
                print("\n" + "="*60)
                print("🎉 FINAL RESULTS SUMMARY")
                print("="*60)
                
                print(f"⏱️  Total execution time: {total_time:.2f} minutes")
                print(f"📊 Dataset size: {len(graph_data_list)} graphs from {len(subjects)} subjects")
                print(f"🔧 Graph structure: {num_sensors} sensors, {num_features} features per node")
                
                print(f"\n🎯 TASK 1 - Activity Recognition:")
                print(f"   • Test Accuracy: {activity_accuracy:.4f}")
                print(f"   • Test F1-Score: {activity_f1:.4f}")
                print(f"   • Improved Accuracy: {improved_activity_accuracy:.4f}")
                print(f"   • Improved Accuracy: {improved_activity_f1:.4f}")
                print(f"   • Activities: {list(activity_map.values())}")
                
                print(f"\n🚨 TASK 2 - Anomaly Detection:")
                print(f"   • Baseline Anomaly Rate: {sum(anomalies)/len(anomalies)*100:.1f}%")
                print(f"   • Baseline Threshold: {threshold:.6f}")
                print(f"   • Baseline Total Anomalies: {sum(anomalies)}/{len(anomalies)}" if anomalies else "   • Total Anomalies: 0/0")


                print(f"   • Improved Anomaly Rate: {sum(improved_reconstruction_errors > threshold) / len(improved_reconstruction_errors) * 100:.2f}%")
                print(f"   • Improved Threshold (P95): {improved_threshold:.6f}")
                print(f"   • Improved Total Anomalies: {sum(improved_anomalies)}/{len(improved_anomalies)}" if improved_anomalies else "   • Total Anomalies: 0/0")
                print(f"   • Total Anomalies: {sum(anomalies)}/{len(anomalies)}")
                
                # Activity-wise performance
                activity_performance = {}
                for activity, idx in activity_to_index.items():
                    # Baseline model
                    mask_base = np.array(labels) == idx
                    acc_base = (np.array(preds)[mask_base] == idx).mean() if mask_base.sum() > 0 else None

                    # Improved model
                    mask_improved = np.array(improved_labels) == idx
                    acc_improved = (np.array(improved_preds)[mask_improved] == idx).mean() if mask_improved.sum() > 0 else None

                    activity_performance[activity] = (acc_base, acc_improved)

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

                # Print results
                print(f"\n📈 Activity-wise Accuracy Comparison:")
                print("   Activity          | Baseline  | Improved ")
                print("   ------------------|-----------|----------")
                for activity, (acc_base, acc_improved) in sorted(activity_performance.items()):
                    acc_base_str = f"{acc_base:.4f}" if acc_base is not None else "N/A"
                    acc_improved_str = f"{acc_improved:.4f}" if acc_improved is not None else "N/A"
                    print(f"   {activity:<18} | {acc_base_str:<9} | {acc_improved_str:<8}")
            
                print("\n✅ Pipeline completed successfully!")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    results_df = pd.DataFrame(results_table)
    print("\n📊 Window/Stride Evaluation Summary:")
    print(results_df.to_string(index=False))

    # Optionally save as CSV
    results_df.to_csv("adl_results.csv", index=False)

if __name__ == "__main__":
    results = main()