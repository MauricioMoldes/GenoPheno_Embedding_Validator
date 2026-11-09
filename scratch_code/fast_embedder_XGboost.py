import cudf
import cugraph
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected
import xgboost as xgb
import cupy as cp
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. Load the real Chromosome Graph 
# ==========================================
# Replace this URL/path with the actual link to your chr6 data
data_file = "path_or_url_to_your_chr6_edgelist.tsv"

# Load into GPU memory. Adjust `sep` and `names` based on your file's format.
# Assuming a standard format: [source_node, target_node, contact_frequency]
df = cudf.read_csv(data_file, sep='\t', names=['src', 'dst', 'weight'])

G = cugraph.Graph(directed=False)

# renumber=True is CRITICAL here. It safely maps arbitrary genomic bin IDs 
# (e.g., 1500000, 1550000) to contiguous integers (0, 1, 2...) for PyTorch.
G.from_cudf_edgelist(df, source='src', destination='dst', renumber=True)

num_nodes = G.number_of_vertices()
print(f"Loaded chr6 graph with {num_nodes} nodes and {G.number_of_edges()} edges.")

# ---------------------------------------------------------
# Create dummy labels (Replace this with your actual labels!)
# ---------------------------------------------------------
# If you have actual node labels (e.g., chromatin states, gene expression active/inactive), 
# load them via cudf.read_csv, join them to G.renumber_map, and convert to CuPy.
y = cp.random.choice(cp.array([0, 1], dtype=cp.int32), size=num_nodes)


# ==========================================
# 2. Extract edges for PyTorch Geometric
# ==========================================
edges = G.view_edge_list()

# Move from cuDF -> CuPy -> DLPack -> PyTorch (keeps data strictly on GPU)
src_tensor = torch.from_dlpack(edges['src'].to_cupy())
dst_tensor = torch.from_dlpack(edges['dst'].to_cupy())

edge_index = torch.stack([src_tensor, dst_tensor]).to(torch.long)
edge_index = to_undirected(edge_index)

# ==========================================
# 3. Generate Graph Embeddings (PyG Node2Vec)
# ==========================================
model = Node2Vec(
    edge_index, 
    embedding_dim=64, 
    walk_length=30, 
    context_size=10, 
    walks_per_node=200,
    num_negative_samples=1, 
    p=1.0, 
    q=1.0, 
    sparse=True
).to(device)

loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

print("Training PyG Node2Vec model...")
model.train()
epochs = 5 
for epoch in range(1, epochs + 1):
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch: {epoch:02d}, Loss: {total_loss / len(loader):.4f}")

# ==========================================
# 4. Create Feature Matrix (X) strictly on GPU
# ==========================================
model.eval()
with torch.no_grad():
    X_tensor = model().detach()
    # Zero-copy transfer from PyTorch to CuPy via DLPack
    X = cp.from_dlpack(torch.utils.dlpack.to_dlpack(X_tensor))

# ==========================================
# 5. Train & Evaluate XGBoost on GPU
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    tree_method='hist',
    device='cuda'
)
xgb_model.fit(X_train, y_train)

preds = xgb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
