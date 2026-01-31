import os
import pickle
import dgl
import math
from matplotlib.pyplot import get
import torch
import networkx as nx
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from torch_geometric.nn import MetaPath2Vec


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h ):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            #print(len(node_dict), len(edge_dict))
            for srctype, etype, dsttype in G.canonical_etypes:
                #print(f"Processing edge type {etype} between {srctype} (source) and {dsttype} (target)")
                sub_graph = G[srctype, etype, dsttype]
                #print(f"Edges for internal_call: {G.edges(etype='internal_call')}")
                canonical_etype = (srctype, etype, dsttype)
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[canonical_etype]
                #print(e_id)

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)
                #print(k.shape)
                #print(q.shape)
                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v
                
                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_scores = sub_graph.edata['t']
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)
            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                #print(alpha)
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                #t = F.elu(t)
                #print(t)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class HGTVulNodeClassifier(nn.Module):
    def __init__(self, compressed_global_graph_path, semantic_embed_path, feature_extractor=None, node_feature='metapath2vec', hidden_size=128, num_layers=2,num_heads=8, use_norm=True, device='cpu',precomputed_feature_path=None):
        super(HGTVulNodeClassifier, self).__init__()
        self.compressed_global_graph_path = compressed_global_graph_path
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device
        # Get Global graph
        nx_graph = load_hetero_nx_graph(compressed_global_graph_path)
        self.nx_graph = nx_graph
        nx_g_data = generate_hetero_graph_data(nx_graph)
        self.total_nodes = len(nx_graph)

        # Reflect graph data
        self.symmetrical_global_graph_data = reflect_graph(nx_g_data)
        # self.symmetrical_global_graph_data = nx_g_data
        self.number_of_nodes = get_number_of_nodes(nx_graph)
        self.symmetrical_global_graph = dgl.heterograph(self.symmetrical_global_graph_data, num_nodes_dict=self.number_of_nodes, device=device)

        # Get sematic
        # load semantic dict (fp.sol → Tensor(256))
        sem_dict = torch.load(semantic_embed_path) 
        original_D_s = next(iter(sem_dict.values())).shape[0]  # 4096
        semantic_dim = 128  # output dim sau projection

        # Gán embedding cho từng node
        num_nodes = self.symmetrical_global_graph.number_of_nodes()
        sem_feats = torch.zeros(num_nodes, original_D_s)

        for node_id, data in nx_graph.nodes(data=True):
            sem_feats[node_id] = sem_dict.get(node_id, torch.zeros(original_D_s))

        self.raw_sem_feats = sem_feats.to(device)

        # Get Node Labels
        self.node_labels, self.labeled_node_ids, self.label_ids = get_node_label(nx_graph)
        self.node_ids_dict = get_node_ids_dict(nx_graph)
        
        # self.meta_paths = get_symmatrical_metapaths(self.symmetrical_global_graph)
        self.meta_paths = get_length_2_metapath(self.symmetrical_global_graph)
        # Concat the metapaths have the same begin nodetype
        self.full_metapath = {}
        for metapath in self.meta_paths:
            ntype = metapath[0][0]
            if ntype not in self.full_metapath:
                self.full_metapath[ntype] = [metapath]
            else:
                self.full_metapath[ntype].append(metapath)
        self.node_types = set([meta_path[0][0] for meta_path in self.meta_paths])
        self.node_types = list(self.symmetrical_global_graph.ntypes)
        
        # node/edge dictionaries
        self.ntypes_dict = {k: v for v, k in enumerate(self.node_types)}
        self.etypes_dict = {}
        for etype in self.symmetrical_global_graph.canonical_etypes:
            self.etypes_dict[etype] = len(self.etypes_dict)
            self.symmetrical_global_graph.edges[etype].data['id'] = \
                torch.ones(self.symmetrical_global_graph.number_of_edges(etype), 
                        dtype=torch.long, device=device) * self.etypes_dict[etype]

        # Create input node features
        self.node_feature = node_feature
        features = {}
        if precomputed_feature_path is not None:
            # Load precomputed node features từ file .pt
            features = torch.load(precomputed_feature_path)
            self.in_size = next(iter(features.values())).shape[1]  # lấy dim từ 1 node
            print(f"[INFO] Loaded precomputed node features from: {precomputed_feature_path}")
        else:
            features = {}
            if node_feature == 'metapath2vec':
                embedding_dim = 128
                self.in_size = embedding_dim
                for metapath in self.meta_paths:
                    _metapath_embedding = MetaPath2Vec(self.symmetrical_global_graph_data,
                                                    embedding_dim=embedding_dim,
                                                    metapath=metapath, walk_length=50,
                                                    context_size=7, walks_per_node=5,
                                                    num_negative_samples=5,
                                                    num_nodes_dict=self.number_of_nodes,
                                                    sparse=False)
                    ntype = metapath[0][0]
                    if ntype not in features:
                        features[ntype] = _metapath_embedding(ntype).unsqueeze(0)
                    else:
                        features[ntype] = torch.cat((features[ntype], _metapath_embedding(ntype).unsqueeze(0)))
                features = {k: torch.mean(v, dim=0).to(self.device) for k, v in features.items()}

        self.node_input_features = features

  
        self.symmetrical_global_graph = self.symmetrical_global_graph.to(self.device)
        for ntype in self.symmetrical_global_graph.ntypes:
            emb = nn.Parameter(features[ntype], requires_grad = False)
            self.symmetrical_global_graph.nodes[ntype].data['inp'] = emb.to(device)

        # Init Model
        self.gcs = nn.ModuleList()
        self.out_size = 9
        self.num_layers = num_layers
        self.adapt_ws  = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.5)
        for t in range(len(self.ntypes_dict)):
            self.adapt_ws.append(nn.Linear(self.in_size, self.hidden_size))
        for _ in range(self.num_layers):
            self.gcs.append(HGTLayer(self.hidden_size, self.hidden_size, self.ntypes_dict, self.etypes_dict, self.num_heads, use_norm=use_norm))
        self.bilstm = nn.LSTM(self.hidden_size + semantic_dim, self.hidden_size,batch_first=True, bidirectional=True) 
        self.classify = nn.Linear(self.hidden_size * 2, self.out_size)
        
    
    

    def reset_parameters(self):
        for model in self.adapt_ws:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for model in self.gcs:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for layer in self.classify.children():
            if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, node_ids=None):
        h = {}
        hiddens = torch.zeros((self.symmetrical_global_graph.number_of_nodes(), self.hidden_size), device=self.device)
        for ntype in self.symmetrical_global_graph.ntypes:
            n_id = self.ntypes_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](self.symmetrical_global_graph.nodes[ntype].data['inp']))
        for i in range(self.num_layers):
            h = self.gcs[i](self.symmetrical_global_graph, h)
        # for ntype, feat in h.items():
        #     print(f"{ntype} => {feat.shape}", flush=True)

        for ntype, feature in h.items():
            assert len(self.node_ids_dict[ntype]) == feature.shape[0]
            hiddens[self.node_ids_dict[ntype]] = feature
        
        if node_ids is not None:
            node_ids = node_ids.long()
            hiddens = hiddens[node_ids]

        sem = self.raw_sem_feats

        if node_ids is not None:
            sem = sem[node_ids.long()]        # (n_batch, D_s)

        

        # 6) CONCAT fusion
        fused = torch.cat([hiddens, sem], dim=1)  # (n_batch, D_g + D_s)
        #print(f"[FORWARD] hiddens shape: {hiddens.shape}", flush=True)
        #print(f"hiddens : {hiddens}", flush=True)
        #print(f"[FORWARD] fused shape: {fused.shape}", flush=True)
        #print(f"fused : {fused}", flush=True)
     

        output, (h_n, c_n) = self.bilstm(fused.unsqueeze(0)) 
        output = self.dropout(output)
        output = self.classify(output)
        return output
        
        
        
def add_hetero_ids(nx_graph):
    nx_g = nx_graph
    dict_hetero_id = {}

    for node, node_data in nx_g.nodes(data=True):
        #print("add_hetero_ids:    ",node, node_data)
        #print(node_data['node_type'])
        if node_data['node_type'] not in dict_hetero_id:
            dict_hetero_id[node_data['node_type']] = 0
            #print("if   ",dict_hetero_id)
        else:
            dict_hetero_id[node_data['node_type']] += 1
            #print("else    ",dict_hetero_id)
        #print(nx_g.nodes[node])
        nx_g.nodes[node]['node_hetero_id'] = dict_hetero_id[node_data['node_type']]
        #print(nx_g.nodes[node])
    return nx_g


def load_hetero_nx_graph(nx_graph_path):
    nx_graph = nx.read_gpickle(nx_graph_path)
    nx_graph = nx.convert_node_labels_to_integers(nx_graph)
    nx_graph = add_hetero_ids(nx_graph)
    return nx_graph
    
def convert_edge_data_to_tensor(dict_egdes):
    dict_three_cannonical_egdes = dict_egdes
    for key, val in dict_three_cannonical_egdes.items():
        list_source = []
        list_target = []
        for source, target in val:
            list_source.append(source)
            list_target.append(target)
        dict_three_cannonical_egdes[key] = (torch.tensor(list_source), torch.tensor(list_target))
    return dict_three_cannonical_egdes
   
def generate_hetero_graph_data(nx_graph):
    nx_g = nx_graph
    dict_three_cannonical_egdes = dict()
    for source, target, data in nx_g.edges(data=True):
        #print(source, target, data )
        edge_type = data['edge_type']
        source_node_type = nx_g.nodes[source]['node_type']
        target_node_type = nx_g.nodes[target]['node_type']
        three_cannonical_egde = (source_node_type, edge_type, target_node_type)
        #print("three_cannonical_egde:   ",three_cannonical_egde)

        if three_cannonical_egde not in dict_three_cannonical_egdes.keys():
            dict_three_cannonical_egdes[three_cannonical_egde] = [(nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])]
            #print("if",  dict_three_cannonical_egdes  )
        else:
            current_val = dict_three_cannonical_egdes[three_cannonical_egde]
            temp_edge = (nx_g.nodes[source]['node_hetero_id'], nx_g.nodes[target]['node_hetero_id'])
            current_val.append(temp_edge)
            dict_three_cannonical_egdes[three_cannonical_egde] = current_val
            #print("else     ", dict_three_cannonical_egdes)

    dict_three_cannonical_egdes = convert_edge_data_to_tensor(dict_three_cannonical_egdes)
    #print(dict_three_cannonical_egdes)
    return dict_three_cannonical_egdes

def get_node_ids_dict(nx_graph):
    nx_g = nx_graph
    node_ids_dict = {}
    for node_ids, node_data in nx_g.nodes(data=True):
        ntype = node_data['node_type']
        if ntype not in node_ids_dict:
            node_ids_dict[ntype] = [node_ids]
        else:
            node_ids_dict[ntype].append(node_ids)
    return node_ids_dict

def get_node_label(nx_graph):
    """
    Assign each node one of 9 classes:
      0: No Vulnerability
      1: Reentrancy
      2: Arithmetic
      3: Access Control
      4: Front-running
      5: Unchecked External Calls
      6: Denial of Service
      7: Block Timestamp Manipulation
      8: Other Vulnerability
    """
    categories = [
        "No Vulnerability",
        "Reentrancy",
        "Arithmetic",
        "Access Control",
        "Front-running",
        "Unchecked External Calls",
        "Denial of Service",
        "Block Timestamp Manipulation",
        "Other Vulnerability"
    ]
    label_ids = {cat: idx for idx, cat in enumerate(categories)}
    labeled_node_ids = {'valid': [], 'buggy': []}
    node_labels = []

    for node_id, data in nx_graph.nodes(data=True):
        raw = data.get('node_info_vulnerabilities')

        # Normalize into a list of dicts
        if isinstance(raw, list):
            vulns = raw
        elif isinstance(raw, str):
            # treat empty/"None" as no vulnerability
            if raw.lower() == "none" or raw.strip() == "":
                vulns = []
            else:
                vulns = [{"category": raw}]
        else:
            vulns = []

        # Decide category
        if not vulns:
            cat = "No Vulnerability"
        else:
            orig = vulns[0].get('category', None)
            cat = orig if orig in label_ids and orig != "No Vulnerability" else "Other Vulnerability"

        target = label_ids[cat]
        if target == 0:
            labeled_node_ids['valid'].append(node_id)
        else:
            labeled_node_ids['buggy'].append(node_id)
        node_labels.append(target)

    return node_labels, labeled_node_ids, label_ids
# def get_node_label(nx_graph):
#     nx_g = nx_graph
#     node_labels = []
#     label_ids = {'valid': 0}
#     labeled_node_ids = {'buggy': [], 'valid': []}
#     for node_id, node_data in nx_g.nodes(data=True):
#         #print(node_id, node_data)
#         node_type = node_data['node_type']
#         node_label = node_data['node_info_vulnerabilities']
#         target = 0
#         if node_label == 'None' or node_label == None:
#            target = 0
#            labeled_node_ids['valid'].append(node_id)     
#         else:
#             bug_type = node_label[0]['category']
#             if bug_type not in label_ids:
#                 label_ids[bug_type] = len(label_ids)
#             target = label_ids[bug_type]
#             # if bug_type == 'time_manipulation':
#             #     target = 1
#             target = 1
#             labeled_node_ids['buggy'].append(node_id)
#         node_labels.append(target)
#     return node_labels, labeled_node_ids, label_ids   
    
def reflect_graph(nx_g_data):
    symmetrical_data = {}
    for metapath, value in nx_g_data.items():
        if metapath[0] == metapath[-1]:
            symmetrical_data[metapath] = (torch.cat((value[0], value[1])), torch.cat((value[1], value[0])))
        else:
            if metapath not in symmetrical_data.keys():
                symmetrical_data[metapath] = value
            else:
                symmetrical_data[metapath] = (torch.cat((symmetrical_data[metapath][0], value[0])), torch.cat((symmetrical_data[metapath][1], value[1])))
            if metapath[::-1] not in symmetrical_data.keys():
                symmetrical_data[metapath[::-1]] = (value[1], value[0])
            else:
                symmetrical_data[metapath[::-1]] = (torch.cat((symmetrical_data[metapath[::-1]][0], value[1])), torch.cat((symmetrical_data[metapath[::-1]][1], value[0])))
    return symmetrical_data

def get_number_of_nodes(nx_graph):
    nx_g = nx_graph
    number_of_nodes = {}
    for node, data in nx_g.nodes(data=True):
        if data['node_type'] not in number_of_nodes.keys():
            number_of_nodes[data['node_type']] = 1
        else:
            number_of_nodes[data['node_type']] += 1
    return number_of_nodes
    
    
def get_length_2_metapath(symmetrical_global_graph):
    begin_by = {}
    end_by = {}
    for mt in symmetrical_global_graph.canonical_etypes:
        if mt[0] not in begin_by:
            begin_by[mt[0]] = [mt]
        else:
            begin_by[mt[0]].append(mt)
        if mt[-1] not in end_by:
            end_by[mt[-1]] = [mt]
        else:
            end_by[mt[-1]].append(mt)
    metapath_list = []
    for mt_0 in symmetrical_global_graph.canonical_etypes:
        source = mt_0[0]
        dest = mt_0[-1]
        if source == dest:
            metapath_list.append([mt_0])
        first_metapath = [mt_0]
        if dest in begin_by:
            for mt_1 in begin_by[dest]:
                if mt_1 != mt_0 and mt_1[-1] == source:
                    second_metapath = first_metapath + [mt_1]
                    metapath_list.append(second_metapath)
    return metapath_list
    
def get_node_vul_ids(graph):
    node_ids = []
    for node_id, node_data in graph.nodes(data=True):
        node_vuln = node_data.get('node_info_vulnerabilities')
        if node_vuln not in [None, "None"]:
            node_ids.append(node_id)
    return node_ids
    
def get_node_non_vul_ids(graph):
    node_ids = []
    for node_id, node_data in graph.nodes(data=True):
        node_vuln = node_data.get('node_info_vulnerabilities')
        if node_vuln in [None, "None"]:
            node_ids.append(node_id)
    return node_ids





      
   






