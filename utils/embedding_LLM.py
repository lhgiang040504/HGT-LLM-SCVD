import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch import nn


# Load model
from huggingface_hub import snapshot_download
import os

# Tải mô hình về thư mục cụ thể
local_dir = "./qwen_model"

print(".....", flush=True)
snapshot_download(
    repo_id="Qwen/CodeQwen1.5-7B-Chat",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # để không symlink nếu bạn đang dùng hệ thống file hạn chế
)
print(".. model:", local_dir, flush=True)
print("running", flush=True)

logging.set_verbosity_info()

local_model_path = "./qwen_model"

print("🚀 Loading model từ local...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/CodeQwen1.5-7B-Chat",
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

print("running1", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
print(device)

projector = nn.Sequential(
    nn.Linear(4096, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128)
).to(device)
projector.eval()

graph_path = "./cfg_cg_compressed_graphs.gpickle"
G = nx.read_gpickle(graph_path)
node_embeddings = {}


for node_id, data in G.nodes(data=True):
    if "source_file" not in data or "node_source_code_lines" not in data:
        continue

    source_file = data["source_file"]
    source_lines = data["node_source_code_lines"]
    if not source_lines or not isinstance(source_lines, list):
        continue

    parts = source_file.split('/')
    if source_file.startswith("/home/kali/"):
        sol_path = os.path.join("./source/sbcurated",*parts[-3:])
    else:
        sol_path = None

    if not sol_path or not os.path.exists(sol_path):
        
        print(sol_path)
        continue

    if not sol_path or not os.path.exists(sol_path):
        print(f"{sol_path} -- lỗi không tồn tại file")
        continue

    with open(sol_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Node lỗi: dòng code vượt số dòng file
    if max(source_lines) > len(lines):
        print(f"{source_file} -- lỗi dòng: source_lines {source_lines} > file {len(lines)} dòng")
        continue

    start_line = max(min(source_lines) - 1, 0)
    end_line = min(max(source_lines) + 1, len(lines))
    code_snippet = "".join(lines[start_line:end_line])

    # Tokenize and get embedding
    inputs = tokenizer(code_snippet, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device).long()
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        token_embeddings = last_hidden.squeeze(0)

    mean_emb = token_embeddings.mean(dim=0).unsqueeze(0).to(device).float()  # (1, 4096)
    proj_emb = projector(mean_emb).squeeze(0).cpu()  # (128,)
    node_embeddings[node_id] = proj_emb
    print(f"Node {node_id} embedding (shape={proj_emb.shape}): {proj_emb[:5]}...", flush=True)

torch.save(node_embeddings, "./node_semantic_embeddings_sb.pt")
