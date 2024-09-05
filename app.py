import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from flask import Flask, request, jsonify
import torch
import re
import torch
from solidity_parser import parser
from torch_geometric.data import Data
import json
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



# Function to parse Solidity file and return its AST
def parse_solidity_file(file_path):
    try:
        source_unit = parser.parse(file_path, loc=True)
        return source_unit
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    
# Function to convert JSON files
def convert_json_files(data):
    data = data.replace("'", '"').replace("False", "false").replace("True", "true").replace("None", "null")
    return data  

# Function to combine string literals
def combine_string_literals(text):
    pattern = re.compile(r'"type":\s*"stringLiteral",\s*"value":\s*"([^"]*)"\s*(("([^"]*)"\s*)*)')
    def combine_values(match):
        combined_value = match.group(1)
        additional_values = re.findall(r'"([^"]*)"', match.group(2))
        combined_value += ' ' + ' '.join(additional_values)
        combined_value = combined_value.replace('\n', ' ').strip()
        return f'"type": "stringLiteral", "value": "{combined_value}"'
    cleaned_text = pattern.sub(combine_values, text)
    return cleaned_text

# Function to process and save the cleaned JSON file
def process_file(data):
    data = combine_string_literals(data)
    return data

# Functions to process the AST into a graph
def extract_nodes_edges(ast):
    nodes, edges = [], []
    def traverse(node, parent_index=None):
        node_index = len(nodes)
        nodes.append(node)
        if parent_index is not None:
            edges.append((parent_index, node_index))
        for key, value in node.items():
            if isinstance(value, dict):
                traverse(value, node_index)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        traverse(item, node_index)
    traverse(ast)
    return nodes, edges

def create_node_features(nodes):
    features = []
    for node in nodes:
        node_type = node.get('type', 'Unknown')
        feature_vector = one_hot_encode_node_type(node_type)
        features.append(feature_vector)
    return torch.tensor(features, dtype=torch.float)

def one_hot_encode_node_type(node_type):
    types = ['PragmaDirective', 'ContractDefinition', 'FunctionDefinition', 'VariableDeclaration', 'BinaryOperation', 'Unknown']
    vector = [0] * len(types)
    if node_type in types:
        vector[types.index(node_type)] = 1
    return vector

def create_edge_index(edges):
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# Function to process AST into graph data
def process_ast(ast):
    nodes, edges = extract_nodes_edges(ast)
    node_features = create_node_features(nodes)
    edge_index = create_edge_index(edges)
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data




app = Flask(__name__)
CORS(app)

# Load the GNN model
gnn_model_path = "gnn_model.pth"
gnn_model = None

# Load your service account credentials
SERVICE_ACCOUNT_FILE = 'apikeys.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

drive_service = build('drive', 'v3', credentials=credentials)

def load_models():
    global gnn_model
    # Load the GNN model
    dummy_data = torch.randn(1, 6)  # Adjust according to your input dimensions
    gnn_model = GNN(num_node_features=dummy_data.size(1), hidden_dim=64, num_classes=2)
    gnn_model.load_state_dict(torch.load(gnn_model_path))
    gnn_model.eval()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    
    if 'content' not in data:
        return jsonify({"error": "No file content"}), 400
    
    file_content = data['content']

    # Process the Solidity code
    ast = parse_solidity_file(file_content)
    if not ast:
        return jsonify({"error": "Error parsing Solidity file"}), 500
    
    ast = str(ast)
    ast = convert_json_files(ast)
    ast = process_file(ast)
    ast = json.loads(ast)

    new_graph = process_ast(ast)

    # GNN Prediction
    with torch.no_grad():
        gnn_out = gnn_model(new_graph)
        gnn_pred = gnn_out.argmax(dim=1).item()
        gnn_vulnerability_status = "Vulnerable" if gnn_pred == 1 else "Not Vulnerable"

    return jsonify({
        'gnn_prediction': gnn_vulnerability_status
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    # Specify the folder ID where you want to save the file
    folder_id = '1EhSjEstNd4y3vBngG5GJ6Ggo8heYyV9q'
    
    # Save file to Google Drive
    file_metadata = {
        'name': file.filename,
        'parents': [folder_id]  # Specify the folder ID here
    }
    media = MediaInMemoryUpload(file.read(), mimetype=file.mimetype)

    file_drive = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    return jsonify({"file_id": file_drive.get('id')}), 200


if __name__ == '__main__':
    load_models()
    app.run(debug=True)