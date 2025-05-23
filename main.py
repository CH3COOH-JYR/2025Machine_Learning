import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import re
import ast # For safely evaluating explain_result string
import math

# --- Configuration ---
# TODO: 请务必修改以下文件路径为你的实际路径
TRAIN_FILE = 'data/train_data.json'
TEST_FILE = 'data/test_data.json' # 你的待预测样本文件名
COLUMN_STATS_FILE = 'data/column_min_max_vals.csv'

# TODO: 请修改为你的姓名和学号
USER_NAME = "任宣宇" # 替换为你的姓名
USER_ID = "2023202303" # 替换为你的学号
OUTPUT_FILE_TEMPLATE = '预测结果_{name}_{id}.csv'

# --- Feature Engineering Parameters ---
MAX_PREDICATES = 10 # 每个查询处理的最大谓词数量
MAX_JOINS = 5     # 每个查询处理的最大连接数量
# Note: MAX_TABLES is implicitly handled by the size of TABLE_TO_IDX for multi-hot table_feature_vec

# --- Global Dictionaries & Lists for Encodings ---
TABLE_TO_IDX, COLUMN_TO_IDX, OPERATOR_TO_IDX, JOIN_KEY_TO_IDX = {}, {}, {}, {}
IDX_TO_TABLE, IDX_TO_COLUMN, IDX_TO_OPERATOR, IDX_TO_JOIN_KEY = [], [], [], []
NODE_TYPE_ENCODING_LIST = []
COLUMN_STATS_DICT_GLOBAL = {} # To store loaded column stats

# --- Feature Vector Segment Dimensions (for Transformer model input) ---
# These will be determined after vocabs are built in preprocess_data
# These are the dimensions *before* projection to d_model
SEG_DIMS = {
    "tables": 0,
    "predicate_atomic": 0, # Dimension of ONE encoded predicate
    "join_atomic": 0,      # Dimension of ONE encoded join
    "plan_node_type": 0,
    "plan_numeric": 4      # startup_cost, total_cost, plan_rows, plan_width
}

# --- 1. Data Loading ---
def load_json_data_robust(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                f.seek(0)
                for line_num, line_content in enumerate(f):
                    line_strip = line_content.strip()
                    if not line_strip: continue
                    try:
                        data.append(json.loads(line_strip))
                    except json.JSONDecodeError as e_line:
                        print(f"Skipping line {line_num+1} in '{file_path}' (JSON Lines attempt) due to error: {e_line}. Line: '{line_strip[:150]}...'")
        if data:
            print(f"Successfully loaded {len(data)} records from '{file_path}'.")
        else:
            print(f"Warning: No data loaded from '{file_path}'.")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with file '{file_path}': {e}")
        return None

def load_column_stats(file_path):
    global COLUMN_STATS_DICT_GLOBAL
    try:
        stats_df = pd.read_csv(file_path)
        COLUMN_STATS_DICT_GLOBAL = {row['name']: row for _, row in stats_df.iterrows()}
        print(f"Loaded column stats from '{file_path}'. Found {len(COLUMN_STATS_DICT_GLOBAL)} entries.")
        return True
    except FileNotFoundError:
        print(f"ERROR: Column stats file not found at '{file_path}'")
        return False
    except Exception as e:
        print(f"Error loading column_stats: {e}")
        return False


# --- 2. Feature Engineering (Parser, Encoders, Normalizers) ---
def safe_json_loads_for_plan(plan_str):
    try:
        return json.loads(plan_str)
    except json.JSONDecodeError:
        try:
            corrected_str = plan_str.replace(": True", ": true").replace(":true", ": true")
            corrected_str = corrected_str.replace(": False", ": false").replace(":false", ": false")
            corrected_str = corrected_str.replace(": None", ": null").replace(":None", ": null")
            
            # Careful with indiscriminate single quote replacement if values can contain them
            # This aims for keys and JSON structure single quotes.
            # A more robust way for keys: re.sub(r"(\s*[\{\,]\s*)'([^']+)':", r'\1"\2":', corrected_str)
            # For now, a general replace, assuming plan strings are somewhat controlled.
            corrected_str = corrected_str.replace("'", '"')
            return json.loads(corrected_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(plan_str)
            except Exception:
                return None

def extract_plan_features(plan_json_str):
    plan_data = safe_json_loads_for_plan(plan_json_str)
    if plan_data is None:
        return {'node_type': 'ErrorParse', 'startup_cost': 0.0, 'total_cost': 0.0,
                'plan_rows': 0.0, 'plan_width': 0.0}, None

    if 'QUERY PLAN' in plan_data and isinstance(plan_data['QUERY PLAN'], list) and len(plan_data['QUERY PLAN']) > 0:
        root_plan_container = plan_data['QUERY PLAN'][0]
    else:
        root_plan_container = plan_data
    root_plan = root_plan_container.get('Plan', {})

    features = {
        'node_type': root_plan.get('Node Type', 'UnknownNode'), # Ensure no 'Error' string that matches others
        'startup_cost': float(root_plan.get('Startup Cost', 0.0)),
        'total_cost': float(root_plan.get('Total Cost', 0.0)),
        'plan_rows': float(root_plan.get('Plan Rows', 0.0)), # Optimizer's estimate
        'plan_width': float(root_plan.get('Plan Width', 0.0)),
    }
    actual_rows = root_plan.get('Actual Rows', None)
    if actual_rows is not None: actual_rows = float(actual_rows)
    return features, actual_rows

def parse_sql_to_structured_features(sql_query_str, table_aliases_g=None):
    sql_lower = sql_query_str.lower()
    tables, table_aliases = set(), (table_aliases_g if table_aliases_g is not None else {})

    # Improved regex for FROM and JOIN clauses to better handle aliases
    # This will find 'table [AS] alias' patterns
    fc_pattern = r'(?:from|join)\s+((?:\w+\s*(?:as\s+)?\w*\s*(?:,|$)\s*)+)'
    all_table_defs = re.findall(fc_pattern, sql_lower)
    
    extracted_table_segments = []
    for seg_group in all_table_defs:
        extracted_table_segments.extend(seg_group.split(','))

    for seg in extracted_table_segments:
        seg = seg.strip()
        if not seg: continue
        parts = seg.split()
        table_name = parts[0]
        tables.add(table_name)
        alias = parts[-1] if len(parts) > 1 and parts[-2].lower() != 'as' else table_name
        if len(parts) > 1 and parts[-2].lower() == 'as': # table AS alias
             alias = parts[-1]
        elif len(parts) > 1 and parts[-2].lower() != 'as' and parts[0] != parts[-1]: # table alias
             alias = parts[-1]
        else: # no alias or just table name
             alias = table_name

        if alias not in table_aliases: # Prioritize existing global aliases if provided for consistency
             table_aliases[alias] = table_name
        if table_name not in table_aliases: # ensure table name itself maps
             table_aliases[table_name] = table_name


    predicates, joins = [], []
    where_clause_match = re.search(r'where\s+(.+?)(?:\sgroup by|\sorder by|\slimit|\szone|$)', sql_lower, re.DOTALL)
    if where_clause_match:
        conditions_str = where_clause_match.group(1)
        atomic_conditions = re.split(r'\s+and\s+', conditions_str, flags=re.IGNORECASE)

        for cond_orig in atomic_conditions:
            cond = cond_orig.strip()
            join_match = re.match(r'([\w\.]+)\s*=\s*([\w\.]+)', cond)
            filter_match = re.match(r'([\w\.]+)\s*([<>=!]{1,2}|like|not\s+like)\s*(.+)', cond, re.IGNORECASE) # Value is more general

            def resolve_col_alias(col_str_full):
                parts = col_str_full.split('.')
                if len(parts) == 2:
                    a, c = parts[0], parts[1]
                    return f"{table_aliases.get(a, a)}.{c}"
                return col_str_full # Should be table.column for predicates

            if join_match:
                j_col1_full, j_col2_full = join_match.groups()
                col1_resolved = resolve_col_alias(j_col1_full.strip())
                col2_resolved = resolve_col_alias(j_col2_full.strip())
                joins.append(tuple(sorted((col1_resolved, col2_resolved))))
            elif filter_match:
                p_col_full, p_op, p_val_str = filter_match.groups()
                col_resolved = resolve_col_alias(p_col_full.strip())
                p_op = p_op.strip().upper() # Normalize operator
                p_val = p_val_str.strip().strip("'").strip('"') # Remove quotes
                predicates.append((col_resolved, p_op, p_val))
                
    return sorted(list(tables)), predicates, sorted(list(set(joins)))


def build_vocabs_from_parsed_sql(all_items_parsed_sql_data):
    global TABLE_TO_IDX, COLUMN_TO_IDX, OPERATOR_TO_IDX, JOIN_KEY_TO_IDX, SEG_DIMS
    global IDX_TO_TABLE, IDX_TO_COLUMN, IDX_TO_OPERATOR, IDX_TO_JOIN_KEY

    s_tables, s_columns, s_operators, s_join_keys = set(), set(), set(), set()
    for tables, predicates, joins in all_items_parsed_sql_data:
        for table in tables: s_tables.add(table)
        for p_col, p_op, _ in predicates:
            s_columns.add(p_col)
            s_operators.add(p_op)
        for j_col1, j_col2 in joins:
            s_join_keys.add(f"{j_col1}--{j_col2}")

    def create_map_and_list(item_set):
        item_list = sorted(list(item_set))
        item_to_idx = {name: i for i, name in enumerate(item_list)}
        return item_to_idx, item_list

    TABLE_TO_IDX, IDX_TO_TABLE = create_map_and_list(s_tables)
    COLUMN_TO_IDX, IDX_TO_COLUMN = create_map_and_list(s_columns)
    OPERATOR_TO_IDX, IDX_TO_OPERATOR = create_map_and_list(s_operators)
    JOIN_KEY_TO_IDX, IDX_TO_JOIN_KEY = create_map_and_list(s_join_keys)
    
    # Update SEG_DIMS based on vocab sizes
    SEG_DIMS["tables"] = len(TABLE_TO_IDX)
    SEG_DIMS["predicate_atomic"] = len(COLUMN_TO_IDX) + len(OPERATOR_TO_IDX) + 1 # col_oh, op_oh, norm_val
    SEG_DIMS["join_atomic"] = len(JOIN_KEY_TO_IDX) # join_key_oh

    print(f"Built vocabs: {len(TABLE_TO_IDX)} T, {len(COLUMN_TO_IDX)} C, "
          f"{len(OPERATOR_TO_IDX)} O, {len(JOIN_KEY_TO_IDX)} JK.")
    print(f"Atomic feature segment dimensions: Predicate={SEG_DIMS['predicate_atomic']}, Join={SEG_DIMS['join_atomic']}")

def encode_one_hot(item, item_to_idx_map, num_classes_override=None):
    idx = item_to_idx_map.get(item, -1)
    num_classes = num_classes_override if num_classes_override is not None else len(item_to_idx_map)
    one_hot = np.zeros(num_classes, dtype=np.float32)
    if idx != -1 and num_classes > 0 : one_hot[idx] = 1.0
    return one_hot

def normalize_value(val_str, col_name, col_stats_dict_local):
    try:
        val_num = float(val_str)
        if col_name in col_stats_dict_local:
            stats = col_stats_dict_local[col_name]
            min_val, max_val = float(stats['min']), float(stats['max'])
            if pd.isna(min_val) or pd.isna(max_val): return np.array([0.5], dtype=np.float32) # Default if stats are NaN
            if max_val > min_val:
                norm_val = (val_num - min_val) / (max_val - min_val)
                return np.array([np.clip(norm_val, 0.0, 1.0)], dtype=np.float32) # Clip to [0,1]
            return np.array([0.5 if val_num == min_val else 0.0], dtype=np.float32)
        return np.array([0.5], dtype=np.float32)
    except ValueError: # Not a number
        return np.array([0.0], dtype=np.float32) # Special value for string comparison, or could be another default


def preprocess_single_item(item_data, is_train_run):
    """ Preprocesses a single data item into its feature segments. """
    global NODE_TYPE_ENCODING_LIST, COLUMN_STATS_DICT_GLOBAL, SEG_DIMS
    
    # 1. SQL Features
    tables, predicates_parsed, joins_parsed = parse_sql_to_structured_features(item_data['query'])
    
    # Table features (multi-hot)
    table_feat_vec = np.zeros(SEG_DIMS["tables"], dtype=np.float32)
    for table_name in tables:
        if table_name in TABLE_TO_IDX: table_feat_vec[TABLE_TO_IDX[table_name]] = 1.0
    
    # Predicate features (list of atomic predicate vectors)
    atomic_predicate_vectors = []
    for p_col, p_op, p_val in predicates_parsed[:MAX_PREDICATES]:
        col_vec = encode_one_hot(p_col, COLUMN_TO_IDX)
        op_vec = encode_one_hot(p_op, OPERATOR_TO_IDX)
        val_norm = normalize_value(p_val, p_col, COLUMN_STATS_DICT_GLOBAL)
        atomic_predicate_vectors.append(np.concatenate([col_vec, op_vec, val_norm]))
    
    # Padding for predicates
    predicate_pad_vec = np.zeros(SEG_DIMS["predicate_atomic"], dtype=np.float32)
    while len(atomic_predicate_vectors) < MAX_PREDICATES:
        atomic_predicate_vectors.append(predicate_pad_vec)
    
    # Join features (list of atomic join vectors)
    atomic_join_vectors = []
    for j_col1, j_col2 in joins_parsed[:MAX_JOINS]:
        join_key = f"{j_col1}--{j_col2}"
        join_vec = encode_one_hot(join_key, JOIN_KEY_TO_IDX)
        atomic_join_vectors.append(join_vec)
        
    # Padding for joins
    join_pad_vec = np.zeros(SEG_DIMS["join_atomic"], dtype=np.float32)
    while len(atomic_join_vectors) < MAX_JOINS:
        atomic_join_vectors.append(join_pad_vec)

    # 2. Plan Features
    plan_features_dict, actual_cardinality = extract_plan_features(item_data['explain_result'])
    
    if is_train_run and actual_cardinality is None: return None # Skip if no target for training
    if plan_features_dict['node_type'] == 'ErrorParse': return None # Skip if plan unparseable

    plan_node_type_vec = encode_one_hot(plan_features_dict['node_type'], 
                                        {name: i for i, name in enumerate(NODE_TYPE_ENCODING_LIST)},
                                        num_classes_override=len(NODE_TYPE_ENCODING_LIST))
    plan_numeric_vec = np.array([
        plan_features_dict['startup_cost'], plan_features_dict['total_cost'],
        plan_features_dict['plan_rows'], plan_features_dict['plan_width']
    ], dtype=np.float32)

    # Assemble all feature segments
    # This structure will be used by the Transformer model to create input sequence
    feature_segments = {
        "tables": table_feat_vec,
        "predicates": atomic_predicate_vectors, # This is a list of vectors
        "joins": atomic_join_vectors,           # This is a list of vectors
        "plan_node_type": plan_node_type_vec,
        "plan_numeric": plan_numeric_vec,
        "query_id": item_data['query_id']
    }
    if is_train_run:
        feature_segments["label"] = np.log1p(actual_cardinality)
        
    return feature_segments


def build_vocabs_and_get_raw_features(raw_data_items, is_train):
    """ Builds vocabs (if train) and extracts features for all items. """
    global NODE_TYPE_ENCODING_LIST
    
    if is_train:
        all_parsed_sql_for_vocab = [parse_sql_to_structured_features(item['query']) for item in raw_data_items]
        build_vocabs_from_parsed_sql(all_parsed_sql_for_vocab)
        
        s_plan_node_types = set()
        for item in raw_data_items:
            plan_f, _ = extract_plan_features(item['explain_result'])
            if plan_f['node_type'] not in ['ErrorParse', 'UnknownNode']:
                 s_plan_node_types.add(plan_f['node_type'])
        NODE_TYPE_ENCODING_LIST = sorted(list(s_plan_node_types))
        SEG_DIMS["plan_node_type"] = len(NODE_TYPE_ENCODING_LIST)
        if not NODE_TYPE_ENCODING_LIST: print("Warning: NODE_TYPE_ENCODING_LIST is empty.")

    all_feature_segments = []
    skipped_count = 0
    for item in raw_data_items:
        processed = preprocess_single_item(item, is_train_run=is_train)
        if processed:
            all_feature_segments.append(processed)
        else:
            skipped_count += 1
    if skipped_count > 0: print(f"Skipped {skipped_count} items during raw feature extraction (is_train={is_train}).")
    return all_feature_segments


# --- 3. PyTorch Dataset & Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50): # max_len for total sequence length
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # .transpose(0, 1) removed: batch_first=True for TransformerEncoder
        self.register_buffer('pe', pe)

    def forward(self, x): # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerCardinalityDataset(Dataset):
    def __init__(self, feature_segments_list, label_scaler_plan_numeric):
        self.feature_segments_list = feature_segments_list
        self.scaler = label_scaler_plan_numeric # For plan_numeric part

    def __len__(self):
        return len(self.feature_segments_list)

    def __getitem__(self, idx):
        item_segments = self.feature_segments_list[idx]
        
        # Normalize the plan_numeric part
        plan_numeric_scaled = self.scaler.transform(item_segments["plan_numeric"].reshape(1, -1)).flatten()

        # Flatten predicates and joins if they are lists of vectors
        # The model's projection layers will handle these inputs
        flat_predicates = np.array(item_segments["predicates"], dtype=np.float32).reshape(MAX_PREDICATES, -1)
        flat_joins = np.array(item_segments["joins"], dtype=np.float32).reshape(MAX_JOINS, -1)

        # Return all segments. The model will combine them into a sequence.
        # This returns raw (but padded) segments. Projection to d_model happens in the model.
        features_dict_out = {
            "tables": torch.FloatTensor(item_segments["tables"]),
            "predicates": torch.FloatTensor(flat_predicates), # [MAX_PREDICATES, pred_atomic_dim]
            "joins": torch.FloatTensor(flat_joins),           # [MAX_JOINS, join_atomic_dim]
            "plan_node_type": torch.FloatTensor(item_segments["plan_node_type"]),
            "plan_numeric": torch.FloatTensor(plan_numeric_scaled)
        }
        
        if "label" in item_segments:
            return features_dict_out, torch.FloatTensor([item_segments["label"]])
        else:
            return features_dict_out


# --- 4. Transformer Model ---
class QueryTransformerEstimator(nn.Module):
    def __init__(self, seg_dims, d_model=128, nhead=4, num_encoder_layers=3, dim_feedforward=256, dropout=0.1):
        super(QueryTransformerEstimator, self).__init__()
        self.d_model = d_model

        # Projection layers for each feature segment to d_model
        self.table_proj = nn.Linear(seg_dims["tables"], d_model)
        self.predicate_proj = nn.Linear(seg_dims["predicate_atomic"], d_model) # Projects each predicate
        self.join_proj = nn.Linear(seg_dims["join_atomic"], d_model)           # Projects each join
        self.plan_node_type_proj = nn.Linear(seg_dims["plan_node_type"], d_model)
        self.plan_numeric_proj = nn.Linear(seg_dims["plan_numeric"], d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=1 + MAX_PREDICATES + MAX_JOINS + 1 + 1 + 1) # CLS + tables + preds + joins + ptype + pnum
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Learnable CLS token

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # Initialize projections and output MLP
        for proj in [self.table_proj, self.predicate_proj, self.join_proj, self.plan_node_type_proj, self.plan_numeric_proj]:
            proj.weight.data.uniform_(-initrange, initrange)
            if proj.bias is not None: proj.bias.data.zero_()
        
        self.output_mlp[0].weight.data.uniform_(-initrange, initrange)
        self.output_mlp[0].bias.data.zero_()
        self.output_mlp[3].weight.data.uniform_(-initrange, initrange)
        self.output_mlp[3].bias.data.zero_()


    def forward(self, features_dict):
        tables_in = features_dict["tables"]         # [B, table_dim]
        predicates_in = features_dict["predicates"] # [B, MAX_PREDICATES, pred_atomic_dim]
        joins_in = features_dict["joins"]           # [B, MAX_JOINS, join_atomic_dim]
        plan_type_in = features_dict["plan_node_type"] # [B, plan_type_dim]
        plan_numeric_in = features_dict["plan_numeric"] # [B, plan_num_dim]
        
        batch_size = tables_in.size(0)

        # Project each segment
        # Each projection results in [B, d_model] or [B, N_segments, d_model]
        tables_emb = self.table_proj(tables_in).unsqueeze(1) # [B, 1, d_model]
        
        # For predicates and joins, project each item in the sequence
        predicates_emb = self.predicate_proj(predicates_in) # [B, MAX_PREDICATES, d_model]
        joins_emb = self.join_proj(joins_in)                # [B, MAX_JOINS, d_model]

        plan_type_emb = self.plan_node_type_proj(plan_type_in).unsqueeze(1) # [B, 1, d_model]
        plan_numeric_emb = self.plan_numeric_proj(plan_numeric_in).unsqueeze(1) # [B, 1, d_model]
        
        # CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [B, 1, d_model]

        # Concatenate into a single sequence for the Transformer
        # Order: CLS, tables, predicates, joins, plan_type, plan_numeric
        seq = torch.cat([cls_tokens, tables_emb, predicates_emb, joins_emb, plan_type_emb, plan_numeric_emb], dim=1)
        # seq shape: [B, 1+1+MAX_PRED+MAX_JOINS+1+1, d_model]
        
        seq_pos_encoded = self.pos_encoder(seq)
        transformer_out = self.transformer_encoder(seq_pos_encoded) # Output shape: [B, seq_len, d_model]
        
        # Use the output of the CLS token for prediction
        cls_out = transformer_out[:, 0, :] # [B, d_model]
        
        output = self.output_mlp(cls_out)
        return output

# --- 5. Training Loop ---
def train_model(model, train_loader, criterion, optimizer, num_epochs=30, device='cpu'):
    model.train()
    print("Starting Transformer model training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_features_dict, batch_labels in train_loader:
            # Move all feature tensors in the dict to device
            for key in batch_features_dict:
                batch_features_dict[key] = batch_features_dict[key].to(device)
            batch_labels = batch_labels.to(device) # Already [B, 1]

            optimizer.zero_grad()
            outputs = model(batch_features_dict)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    print("Training finished.")

# --- 6. Prediction ---
def predict_cardinalities(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_features_dict in test_loader:
            for key in batch_features_dict:
                batch_features_dict[key] = batch_features_dict[key].to(device)
            
            outputs = model(batch_features_dict)
            preds = np.expm1(outputs.cpu().numpy())
            predictions.extend(np.maximum(0, preds.flatten()))
    return predictions

# --- Main Execution ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not load_column_stats(COLUMN_STATS_FILE): exit(1) # Load stats globally
    
    train_raw = load_json_data_robust(TRAIN_FILE)
    test_raw = load_json_data_robust(TEST_FILE)

    if not train_raw or not test_raw:
        print("FATAL: Failed to load train or test data. Exiting.")
        exit(1)

    # Build vocabs and extract all features (SEG_DIMS gets updated here)
    print("Preprocessing training data (building vocabs)...")
    train_feature_segments = build_vocabs_and_get_raw_features(train_raw, is_train=True)
    if not train_feature_segments:
        print("FATAL: No training samples generated after preprocessing. Exiting.")
        exit(1)
    
    print(f"Generated {len(train_feature_segments)} training feature segment groups.")
    print("Global SEG_DIMS after vocab build:", SEG_DIMS)


    print("Preprocessing test data (using existing vocabs)...")
    test_feature_segments = build_vocabs_and_get_raw_features(test_raw, is_train=False)
    if not test_feature_segments:
        print("Warning: No test samples generated. Predictions might be default.")
    else:
        print(f"Generated {len(test_feature_segments)} test feature segment groups.")

    # Scaler for plan_numeric features - fit only on training data
    plan_numeric_train_data = np.array([fs["plan_numeric"] for fs in train_feature_segments if fs])
    scaler_plan_numeric = StandardScaler()
    if plan_numeric_train_data.size > 0:
        scaler_plan_numeric.fit(plan_numeric_train_data)
    else:
        print("Warning: No plan_numeric data from training to fit scaler. Using default (no scaling).")
        # Create a dummy scaler that does nothing if no data
        class DummyScaler:
            def fit(self, X): return self
            def transform(self, X): return X
        scaler_plan_numeric = DummyScaler()


    train_dataset = TransformerCardinalityDataset(train_feature_segments, scaler_plan_numeric)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0) # Smaller batch for Transformer

    # Ensure SEG_DIMS are valid before model init
    if any(v == 0 for k, v in SEG_DIMS.items() if k != "plan_numeric" and k != "plan_node_type"): # Some vocabs might be empty
        # Allow plan_node_type to be zero if no valid nodes found, projection will handle it
        pass # Model projection layers should handle zero-dim inputs if a vocab is empty by chance.
             # Or add a small constant to avoid zero-dim tensors if strictly needed by Linear.
             # For now, nn.Linear(0, d_model) will error. Let's ensure non-zero for projection layers.
    for key in ["tables", "predicate_atomic", "join_atomic", "plan_node_type"]:
        if SEG_DIMS.get(key, 0) == 0:
            print(f"Warning: SEG_DIMS['{key}'] is 0. Setting to 1 to avoid Linear(0,...) error. Feature will be all zeros.")
            SEG_DIMS[key] = 1 # Avoid nn.Linear(0,...)

    model = QueryTransformerEstimator(SEG_DIMS, d_model=64, nhead=4, num_encoder_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # AdamW, smaller LR

    train_model(model, train_loader, criterion, optimizer, num_epochs=40, device=device) # Epochs might need adjustment

    # Predictions
    test_qids_final = [item['query_id'] for item in test_raw] # Get all original QIDs first
    test_predictions_final = [0.0] * len(test_raw) # Default predictions

    if test_feature_segments:
        test_dataset = TransformerCardinalityDataset(test_feature_segments, scaler_plan_numeric)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        print("Predicting on test data...")
        predictions_list = predict_cardinalities(model, test_loader, device=device)
        
        # Map predictions back to original query IDs
        pred_map = {fs["query_id"]: pred for fs, pred in zip(test_feature_segments, predictions_list) if fs}
        for i, qid in enumerate(test_qids_final):
            test_predictions_final[i] = pred_map.get(qid, 0.0) # Use default if QID not in processed map
    else:
        print("No test features were generated; all predictions remain default 0.0.")
        
    output_filename = OUTPUT_FILE_TEMPLATE.format(name=USER_NAME, id=USER_ID)
    output_df = pd.DataFrame({'Query ID': test_qids_final, 'Predicted Cardinality': test_predictions_final})
    output_df['Query ID'] = output_df['Query ID'].astype(int)
    output_df['Predicted Cardinality'] = output_df['Predicted Cardinality'].round(2)

    try:
        output_df.to_csv(output_filename, index=False)
        print(f"Predictions successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        print("Dumping predictions to console:\n", output_df.to_string())
    print("Script finished.")