import streamlit as st
import torch
import numpy as np
from Bio.PDB import PDBParser, is_aa
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import to_undirected


# Define the GCN model
class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Output layer for solubility prediction

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Global average pooling
        return self.fc(x)

# Load your trained model
def load_model(model_path):
    model = GCNModel(num_node_features=20, hidden_channels=16)  # Adjust parameters as needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to parse the PDB file and generate adjacency matrix and node features
def parse_pdb(file_path, threshold=5.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    
    # Extract amino acid residues
    residues = [residue for residue in structure.get_residues() if is_aa(residue)]
    
    # Generate adjacency matrix
    num_residues = len(residues)
    adjacency_matrix = np.zeros((num_residues, num_residues))

    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues):
            if i != j:
                min_distance = min(atom1 - atom2 for atom1 in res1 if atom1.element != 'H' 
                                                        for atom2 in res2 if atom2.element != 'H')
                if min_distance <= threshold:
                    adjacency_matrix[i, j] = 1  # Contact

    # Generate node features (example: one-hot encoding for residue types)
    # For simplicity, let's create dummy features here
    node_features = np.zeros((num_residues, 20))  # Adjusted to 20 features
    for i, res in enumerate(residues):
        # Example: Assigning a random feature vector for each residue (you'll want to replace this with meaningful features)
        node_features[i] = np.random.rand(20)  # Random features (replace with actual feature extraction)

    return adjacency_matrix, node_features, num_residues

# Function for making a prediction
def make_prediction(file_path, model_path):
    # Load the trained model
    model = load_model(model_path)

    # Parse the PDB file to get adjacency matrix and node features
    adjacency_matrix, node_features, num_nodes = parse_pdb(file_path)

    # Create a PyG Data object
    edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long)
    edge_index = to_undirected(edge_index)  # Ensure the graph is undirected
    x = torch.tensor(node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)

    # Make prediction
    with torch.no_grad():
        prediction = model(data)  # No need to add a batch dimension

    # Return the prediction
    return prediction.item()

# Streamlit UI
def main():
    st.title("Protein Solubility Prediction by Sai Nivedh")
    
    uploaded_file = st.file_uploader("Choose a PDB file...", type="pdb")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("uploaded_file.pdb", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the model path
        trained_model_path = "gcn_model.pth"  # Adjust this path
        
        # Indicate model loading progress
        progress_bar = st.progress(0)
        st.text("Loading model...")
        model = load_model(trained_model_path)
        progress_bar.progress(33)

        # Indicate PDB parsing progress
        st.text("Parsing PDB file...")
        adjacency_matrix, node_features, num_nodes = parse_pdb("uploaded_file.pdb")
        progress_bar.progress(66)

        # Indicate prediction progress
        st.text("Making prediction...")
        prediction = make_prediction("uploaded_file.pdb", trained_model_path)
        progress_bar.progress(100)

        st.success(f"Predicted solubility: {prediction:.4f}")

if __name__ == "__main__":
    main()
