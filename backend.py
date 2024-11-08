import requests
import torch
import numpy as np
from Bio.PDB import PDBParser, is_aa
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool
import torch.nn.functional as F
from stmol import showmol
import py3Dmol
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()

# Get NVIDIA API key from environment variable
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Define the GCN model with adjusted dimensions
class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        return self.fc(x)

# Load model function
def load_model(model_path):
    model = GCNModel(num_node_features=20, hidden_channels=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def calculate_distance_matrix(atoms):
    num_atoms = len(atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            distance = np.linalg.norm(atoms[i] - atoms[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

# Extract sequence from PDB
def extract_sequence_from_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    residues = [residue for residue in structure.get_residues() if is_aa(residue)]
    sequence = ''.join([residue.get_resname()[0] for residue in residues])
    return sequence

# Generate adjacency matrix for residues within a threshold distance
def pdb_to_adjacency_matrix(file_path, threshold=5.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    residues = [residue for residue in structure.get_residues() if is_aa(residue)]
    adjacency_matrix = calculate_residue_distance_matrix(residues, threshold)
    return adjacency_matrix

# Calculate residue distance matrix
def calculate_residue_distance_matrix(residues, threshold=5.0):
    num_residues = len(residues)
    adjacency_matrix = np.zeros((num_residues, num_residues))

    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues):
            if i != j:
                min_distance = min(atom1 - atom2 for atom1 in res1 if atom1.element != 'H'
                                                    for atom2 in res2 if atom2.element != 'H')
                if min_distance <= threshold:
                    adjacency_matrix[i, j] = 1
    return adjacency_matrix

# Generate node features
def generate_node_features(sequence):
    feature_dict = {
        'A': [1.8, 0, 89.1], 'C': [2.5, 1, 121.2], 'D': [-3.5, 1, 133.1], 'E': [-3.5, 1, 147.1],
        'F': [2.8, 0, 165.2], 'G': [-0.4, 0, 75.0], 'H': [-3.2, 1, 155.2], 'I': [4.5, 0, 131.2],
        'K': [-3.9, 1, 146.2], 'L': [3.8, 0, 131.2], 'M': [1.9, 0, 149.2], 'N': [-3.5, 1, 132.1],
        'P': [13, 0, 115.1], 'Q': [-3.5, 1, 146.2], 'R': [-4.5, 1, 174.2], 'S': [-0.8, 1, 105.1],
        'T': [-0.7, 1, 119.1], 'V': [4.2, 0, 117.1], 'W': [-3.4, 1, 204.2], 'Y': [-1.3, 1, 181.2]
    }
    node_features = [
        feature_dict.get(aa.upper(), [0] * len(next(iter(feature_dict.values())))) + [0] * (20 - len(feature_dict.get(aa.upper(), [])))
        for aa in sequence
    ]
    return np.array(node_features)

# Fetch PDB from API
def fetch_pdb_from_api(sequence):
    invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",   
        "Accept": "application/json",
    }
    payload = {"sequence": sequence}
    session = requests.Session()
    response = session.post(invoke_url, headers=headers, json=payload)
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP error occurred: {e}")
        return

    response_body = response.json()
    
    if 'pdbs' in response_body and len(response_body['pdbs']) > 0:
        pdb_content = response_body['pdbs'][0]
        with open("output.pdb", "w") as f:
            f.write(pdb_content)
        st.success("PDB file fetched successfully.")
        return pdb_content
    else:
        st.error("PDB content not found in the API response.")

# Render the 3D structure
def render_pdb_structure(pdb_content):
    view = py3Dmol.view(width=800, height=500)
    view.addModel(pdb_content, "pdb")
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view

# Plot heatmap of distance matrix
def pdb_to_heatmap(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    atoms = [atom.get_vector() for atom in structure.get_atoms() if atom.element != 'H']
    distance_matrix = calculate_distance_matrix(atoms)

    fig, ax = plt.subplots(figsize=(5,4))
    cax = ax.imshow(distance_matrix, cmap="hot", interpolation="nearest")
    fig.colorbar(cax, label="Distance (Å)")
    ax.set_title("Protein Distance Matrix Heatmap")
    ax.set_xlabel("Atom index")
    ax.set_ylabel("Atom index")
    return fig

# Plot adjacency matrix as a heatmap
def plot_adjacency_matrix(file_path, threshold=5.0):
    adjacency_matrix = pdb_to_adjacency_matrix(file_path, threshold)

    fig, ax = plt.subplots(figsize=(5,4))
    cax = ax.imshow(adjacency_matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, label="Contact (1 = contact, 0 = no contact)")
    ax.set_title(f"Residue Contact Adjacency Matrix (Threshold = {threshold} Å)")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    return fig

# Main Streamlit app
def main():
    st.set_page_config(page_title="Protein Solubility Predictor", page_icon="🔬🧬", layout="wide")
    st.title("Protein Solubility Predictor")
    sequence_input = st.text_input("Enter Protein Sequence:")

    if st.button("Fetch PDB and Predict"):
        if sequence_input:
            with st.spinner("Fetching PDB file from ESMFold API..."):
                pdb_content = fetch_pdb_from_api(sequence_input)

            if pdb_content:
                model_path = "E:\\Btech\\S3\\bio\\IBS_Project\\gcn_model_2.pth"
                with st.spinner("Loading model..."):
                    model = load_model(model_path)

                with st.spinner("Processing PDB file..."):
                    sequence = extract_sequence_from_pdb("output.pdb")
                    node_features = generate_node_features(sequence)
                    adjacency_matrix = pdb_to_adjacency_matrix("output.pdb", threshold=5.0)
                    data = Data(x=torch.tensor(node_features, dtype=torch.float32),
                                edge_index=torch.tensor(np.array(adjacency_matrix).nonzero(), dtype=torch.long))

                with st.spinner("Running prediction..."):
                    prediction = model(data)
                    st.write(f"Protein Solubility Prediction Score: {prediction.item():.3f}")

                with st.spinner("Rendering 3D structure..."):
                    view = render_pdb_structure(pdb_content)
                    showmol(view, width=800, height=500)

                with st.spinner("Generating heatmap of distance matrix..."):
                    fig1 = pdb_to_heatmap("output.pdb")
                    st.pyplot(fig1)

                with st.spinner("Generating adjacency matrix heatmap..."):
                    fig2 = plot_adjacency_matrix("output.pdb", threshold=5.0)
                    st.pyplot(fig2)

if __name__ == "__main__":
    main()
