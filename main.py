# Final Optimized Code with Combined Callback, pdb-name Component, Phi/Psi Representations, and Atom Labeling

import dash
from dash import dcc, html, Input, Output, State
import base64
import os
import io
import py3Dmol
import plotly.graph_objects as go
import numpy as np
from Bio import PDB
from Bio.PDB import PPBuilder

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the Flask server for deployments

# Define the directory to store uploaded PDB files
UPLOAD_DIRECTORY = "uploaded_pdbs"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Default PDB file path
DEFAULT_PDB = '5xgu.pdb'

# Load the default PDB file
try:
    with open(DEFAULT_PDB, 'r') as f:
        default_pdb_data = f.read()
except FileNotFoundError:
    default_pdb_data = ""
    print(f"Warning: {DEFAULT_PDB} not found. Please ensure the file exists in the directory.")

# Helper Functions

def parse_pdb(pdb_content):
    """
    Parses PDB data from a string and extracts phi/psi angles, secondary structures, and atom coordinates.

    Parameters:
        pdb_content (str): The content of a PDB file as a string.

    Returns:
        tuple: (phi_psi_data, all_residues, residue_info)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('uploaded', io.StringIO(pdb_content))

    # Extract secondary structures
    secondary_structure = {}
    for line in pdb_content.split('\n'):
        if line.startswith('HELIX'):
            chain_id = line[19]
            try:
                start_res_num = int(line[21:25].strip())
                end_res_num = int(line[33:37].strip())
                for res_num in range(start_res_num, end_res_num + 1):
                    secondary_structure[(chain_id, res_num)] = 'Helix'
            except ValueError:
                continue  # Skip lines with parsing issues
        elif line.startswith('SHEET'):
            chain_id = line[21]
            try:
                start_res_num = int(line[22:26].strip())
                end_res_num = int(line[33:37].strip())
                for res_num in range(start_res_num, end_res_num + 1):
                    secondary_structure[(chain_id, res_num)] = 'Sheet'
            except ValueError:
                continue  # Skip lines with parsing issues

    ppb = PPBuilder()
    phi_psi_data = []
    all_residues = []
    residue_info = []

    for pp in ppb.build_peptides(structure):
        for res in pp:
            chain_id = res.get_parent().get_id()
            res_id = res.get_id()[1]
            # Extract coordinates for main atoms
            atom_coords = {}
            for atom_name in ['N', 'CA', 'C', 'O']:
                if atom_name in res:
                    coord = res[atom_name].get_coord()
                    atom_coords[atom_name] = coord.tolist()  # Convert to list for serialization
                else:
                    atom_coords[atom_name] = None  # Handle missing atoms
            all_residues.append({
                'chain': chain_id,
                'index': res_id,
                'atom_coords': atom_coords  # Store atom coordinates
            })

    for pp in ppb.build_peptides(structure):
        for i, res in enumerate(pp):
            phi, psi = pp.get_phi_psi_list()[i]
            if None not in [phi, psi]:
                res_name = res.get_resname()
                res_id = res.get_id()[1]
                chain_id = res.get_parent().get_id()
                sec_struct = secondary_structure.get((chain_id, res_id), 'Unknown')

                # Retrieve atom coordinates
                atom_coords = {}
                for atom_name in ['N', 'CA', 'C']:
                    if atom_name in res:
                        coord = res[atom_name].get_coord()
                        atom_coords[atom_name] = coord.tolist()
                    else:
                        atom_coords[atom_name] = None  # Handle missing atoms

                phi_psi_data.append({
                    'residue': res_name,
                    'index': res_id,
                    'chain': chain_id,
                    'phi': phi,
                    'psi': psi,
                    'sec_struct': sec_struct,
                    'atom_coords': atom_coords  # Store atom coordinates
                })

                residue_info.append(f"{res_name} {res_id} {sec_struct}")

    return phi_psi_data, all_residues, residue_info

def find_residue_position(chain_id, res_id, all_residues):
    """
    Finds the position of a residue in the all_residues list.

    Parameters:
        chain_id (str): Chain identifier.
        res_id (int): Residue number.
        all_residues (list): List of all residues.

    Returns:
        int or None: Index position if found, else None.
    """
    for i, res in enumerate(all_residues):
        if res['chain'] == chain_id and res['index'] == res_id:
            return i
    return None

def generate_3d_viewer(full_pdb_data, selected_residue=None, all_residues=None):
    """
    Generates the HTML for the 3D molecular viewer with optional atom annotations.

    Parameters:
        full_pdb_data (str): The full PDB data as a string.
        selected_residue (dict or None): The selected residue data.
        all_residues (list or None): List of all residues.

    Returns:
        str: HTML content for the 3D viewer.
    """
    viewer = py3Dmol.view(width=800, height=600)
    viewer.addModel(full_pdb_data, "pdb")

    # Apply cartoon style to the entire structure
    viewer.setStyle({'cartoon': {'color': 'spectrum'}})

    if selected_residue and all_residues:
        pos = find_residue_position(selected_residue['chain'], selected_residue['index'], all_residues)
        if pos is not None:
            # Highlight selected residue
            viewer.setStyle({'resi': str(selected_residue['index']), 'chain': selected_residue['chain']},
                           {'stick': {'color': 'red', 'radius': 0.3}})

            # Highlight previous residue
            if pos > 0:
                prev_res = all_residues[pos - 1]
                viewer.setStyle({'resi': str(prev_res['index']), 'chain': prev_res['chain']},
                               {'stick': {'color': 'blue', 'radius': 0.2}})

            # Highlight next residue
            if pos < len(all_residues) - 1:
                next_res = all_residues[pos + 1]
                viewer.setStyle({'resi': str(next_res['index']), 'chain': next_res['chain']},
                               {'stick': {'color': 'green', 'radius': 0.2}})

            # Label main atoms
            for atom_name in ['N', 'CA', 'C']:
                coord = selected_residue['atom_coords'].get(atom_name)
                if coord:
                    viewer.addSphere({
                        'center': {'x': coord[0], 'y': coord[1], 'z': coord[2]},
                        'radius': 0.2,
                        'color': 'yellow'
                    })
                    viewer.addLabel(
                        atom_name,
                        {
                            'position': {'x': coord[0], 'y': coord[1], 'z': coord[2]},
                            'backgroundColor': 'white',
                            'fontSize': 10,
                            'fontColor': 'black'
                        }
                    )

            # Represent Phi and Psi angles
            # Phi: C(i-1) - N(i) - CA(i) - C(i)
            if pos > 0:
                prev_res = all_residues[pos - 1]
                c_prev = prev_res['atom_coords'].get('C')
                n = selected_residue['atom_coords'].get('N')
                ca = selected_residue['atom_coords'].get('CA')
                c = selected_residue['atom_coords'].get('C')
                if all([c_prev, n, ca, c]):
                    viewer.addCylinder({
                        'start': {'x': c_prev[0], 'y': c_prev[1], 'z': c_prev[2]},
                        'end': {'x': c[0], 'y': c[1], 'z': c[2]},
                        'color': 'green',
                        'radius': 0.1
                    })
                    # Label for Phi
                    mid_phi = [(c_prev[0] + c[0]) / 2, (c_prev[1] + c[1]) / 2, (c_prev[2] + c[2]) / 2]
                    viewer.addLabel(
                        f"Phi: {np.degrees(selected_residue['phi']):.1f}°",
                        {
                            'position': {'x': mid_phi[0], 'y': mid_phi[1], 'z': mid_phi[2]},
                            'backgroundColor': 'white',
                            'fontSize': 10,
                            'fontColor': 'black'
                        }
                    )

            # Psi: N(i) - CA(i) - C(i) - N(i+1)
            if pos < len(all_residues) - 1:
                next_res = all_residues[pos + 1]
                n_next = next_res['atom_coords'].get('N')
                n = selected_residue['atom_coords'].get('N')
                ca = selected_residue['atom_coords'].get('CA')
                c = selected_residue['atom_coords'].get('C')
                if all([n, ca, c, n_next]):
                    viewer.addCylinder({
                        'start': {'x': n[0], 'y': n[1], 'z': n[2]},
                        'end': {'x': n_next[0], 'y': n_next[1], 'z': n_next[2]},
                        'color': 'orange',
                        'radius': 0.1
                    })
                    # Label for Psi
                    mid_psi = [(n[0] + n_next[0]) / 2, (n[1] + n_next[1]) / 2, (n[2] + n_next[2]) / 2]
                    viewer.addLabel(
                        f"Psi: {np.degrees(selected_residue['psi']):.1f}°",
                        {
                            'position': {'x': mid_psi[0], 'y': mid_psi[1], 'z': mid_psi[2]},
                            'backgroundColor': 'white',
                            'fontSize': 10,
                            'fontColor': 'black'
                        }
                    )

            # Zoom into the selected residue
            viewer.zoomTo({'resi': str(selected_residue['index']), 'chain': selected_residue['chain']})

    viewer.render()
    viewer_html = viewer._make_html()
    return viewer_html

def create_ramachandran_plot(phi_psi_data, residue_info):
    """
    Creates an interactive Ramachandran plot using Plotly.

    Parameters:
        phi_psi_data (list): List of dictionaries containing phi and psi angles.
        residue_info (list): List of strings for hover information.

    Returns:
        plotly.graph_objects.Figure: The Ramachandran plot figure.
    """
    phi_psi = np.array([[item['phi'], item['psi']] for item in phi_psi_data])

    # Define color mapping based on secondary structure
    secondary_structure_colors = {
        'Helix': 'blue',
        'Sheet': 'yellow',
        'Unknown': 'grey'
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=phi_psi[:, 0] * 180 / np.pi,  # Phi in degrees
        y=phi_psi[:, 1] * 180 / np.pi,  # Psi in degrees
        mode='markers',
        marker=dict(
            color=[secondary_structure_colors.get(item['sec_struct'], 'grey') for item in phi_psi_data],
            size=6,
            opacity=0.7  # Slight opacity for better visualization
        ),
        text=residue_info,  # Hover text
        customdata=[{'index': item['index'], 'chain': item['chain']} for item in phi_psi_data],  # For callbacks
        hoverinfo='text'
    ))

    # Add legend for secondary structures
    for sec_struct, color in secondary_structure_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=sec_struct
        ))

    fig.update_layout(
        title='Ramachandran Plot',
        xaxis_title='Phi (degrees)',
        yaxis_title='Psi (degrees)',
        showlegend=True,
        xaxis=dict(range=[-180, 180]),
        yaxis=dict(range=[-180, 180]),
        template='plotly_dark'
    )

    return fig

# Initial Processing for Default PDB
phi_psi_data_default, all_residues_default, residue_info_default = parse_pdb(default_pdb_data)
ramachandran_fig_default = create_ramachandran_plot(phi_psi_data_default, residue_info_default)
viewer_html_default = generate_3d_viewer(full_pdb_data=default_pdb_data, selected_residue=None, all_residues=all_residues_default)

# App Layout
app.layout = html.Div([
    html.H1("Molecular Visualization Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.H2(
        children=[
            "By: S.Alireza Hashemi ",
            html.A("salireza111.github.io", href="https://salireza111.github.io/", target="_blank")
        ],
        style={'textAlign': 'center', 'marginBottom': '15px'}
    ),
    
    # Upload Button with Loading Indicator
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a PDB File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'marginBottom': '20px'
            },
            multiple=False,
            accept='.pdb'  # Restrict to PDB files
        ),
    ], style={'width': '90%', 'margin': '0 auto'}),
    
    # Added pdb-name component
    html.Div(id='pdb-name', style={'textAlign': 'center', 'marginBottom': '15px', 'fontSize': '20px', 'fontWeight': 'bold'}),
    
    # Ramachandran Plot and 3D Viewer with Loading Indicators
    html.Div([
        # Ramachandran Plot with Loading Spinner
        html.Div([
            dcc.Loading(
                id="loading-plot",
                type="default",
                children=html.Div([
                    dcc.Graph(
                        id='ramachandran-plot',
                        figure=ramachandran_fig_default
                    )
                ])
            )
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # 3D Viewer with Loading Spinner
        html.Div([
            dcc.Loading(
                id="loading-viewer",
                type="default",
                children=html.Div(id='3d-viewer', children=[
                    html.Iframe(srcDoc=viewer_html_default, width='800px', height='600px')
                ])
            )
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Residue Information Panel
    html.Div(id='residue-info', style={'marginTop': '20px', 'fontSize': '18px', 'textAlign': 'center'}),
    
    # Hidden Store to Cache Parsed PDB Data
    dcc.Store(id='pdb-data', storage_type='memory')  # Use 'memory' or 'session' based on preference
], style={'width': '90%', 'margin': '0 auto'})

# Combined Callback to Handle Uploads and Plot Clicks
@app.callback(
    [
        Output('ramachandran-plot', 'figure'),
        Output('3d-viewer', 'children'),
        Output('residue-info', 'children'),
        Output('pdb-data', 'data'),  # Cache parsed PDB data
        Output('pdb-name', 'children')
    ],
    [
        Input('upload-data', 'contents'),
        Input('ramachandran-plot', 'clickData')
    ],
    [
        State('upload-data', 'filename'),
        State('pdb-data', 'data')
    ]
)
def update_visualizations(upload_contents, clickData, upload_filename, cached_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        # No action needed if no triggers
        return [
            dash.no_update,  # ramachandran-plot.figure
            dash.no_update,  # 3d-viewer.children
            "Upload a PDB file to visualize its Ramachandran plot and 3D structure.",
            dash.no_update,  # pdb-data.data
            "Current PDB: 5xgu.pdb"
        ]

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'upload-data' and upload_contents is not None:
        # Handle file upload
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            pdb_content = decoded.decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return [
                dash.no_update,
                dash.no_update,
                "Error: The uploaded file is not a valid PDB text file.",
                dash.no_update,  # pdb-data.data remains unchanged
                dash.no_update
            ]

        # Parse the uploaded PDB content
        phi_psi_data, all_residues, residue_info = parse_pdb(pdb_content)

        if not phi_psi_data:
            return [
                dash.no_update,
                dash.no_update,
                "No valid residues with phi/psi angles found in the uploaded file.",
                dash.no_update,  # pdb-data.data remains unchanged
                dash.no_update
            ]

        # Create Ramachandran Plot
        fig = create_ramachandran_plot(phi_psi_data, residue_info)

        # Generate 3D Viewer HTML without highlighting
        viewer_html = generate_3d_viewer(full_pdb_data=pdb_content, selected_residue=None, all_residues=all_residues)

        # Prepare data to store (cache)
        pdb_store_data = {
            'pdb_content': pdb_content,
            'phi_psi_data': phi_psi_data,
            'all_residues': all_residues,
            'residue_info': residue_info
        }

        # Update PDB Name Display
        pdb_name = f"Current PDB: {upload_filename}"

        return [
            fig,
            html.Div([
                html.Iframe(srcDoc=viewer_html, width='800px', height='600px')
            ]),
            "Upload successful. Please select a residue in the Ramachandran plot to view details.",
            pdb_store_data,  # Update pdb-data.store with PDB content
            pdb_name  # This updates the 'pdb-name' component
        ]

    elif triggered_id == 'ramachandran-plot' and clickData is not None:
        # Handle click on Ramachandran plot
        try:
            residue_info_clicked = clickData['points'][0]['customdata']
            residue_index = residue_info_clicked['index']
            residue_chain = residue_info_clicked['chain']
        except (IndexError, KeyError, TypeError):
            return [
                dash.no_update,
                dash.no_update,
                "Error retrieving residue information from the plot.",
                dash.no_update,  # pdb-data.data remains unchanged
                dash.no_update  # pdb-name.children remains unchanged
            ]

        # Determine the current PDB data being viewed
        if cached_data is not None and 'pdb_content' in cached_data:
            pdb_content = cached_data['pdb_content']
            phi_psi_data_current = cached_data['phi_psi_data']
            all_residues_current = cached_data['all_residues']
            residue_info_current = cached_data['residue_info']
        else:
            # Use default PDB data if no upload has been done
            pdb_content = default_pdb_data
            phi_psi_data_current = phi_psi_data_default
            all_residues_current = all_residues_default
            residue_info_current = residue_info_default

        # Find the selected residue
        selected_residue = next((item for item in phi_psi_data_current if item['index'] == residue_index and item['chain'] == residue_chain), None)

        if selected_residue is None:
            return [
                dash.no_update,
                dash.no_update,
                "Selected residue not found.",
                dash.no_update,  # pdb-data.data remains unchanged
                dash.no_update  # pdb-name.children remains unchanged
            ]

        # Generate updated 3D viewer with highlighted residue
        viewer_html = generate_3d_viewer(full_pdb_data=pdb_content, selected_residue=selected_residue, all_residues=all_residues_current)

        # Prepare residue information text
        residue_text = (
            f"**Residue:** {selected_residue['residue']} {selected_residue['index']}<br>"
            f"**Chain:** {selected_residue['chain']}<br>"
            f"**Secondary Structure:** {selected_residue['sec_struct']}<br>"
            f"**Phi Angle:** {np.degrees(selected_residue['phi']):.1f}°<br>"
            f"**Psi Angle:** {np.degrees(selected_residue['psi']):.1f}°"
        )

        return [
            dash.no_update,  # Ramachandran plot remains unchanged
            html.Div([
                html.Iframe(srcDoc=viewer_html, width='800px', height='600px')
            ]),
            html.Div([
                html.H3("Selected Residue Information"),
                html.P(dcc.Markdown(residue_text))
            ]),
            dash.no_update,  # pdb-data.data remains unchanged
            dash.no_update  # pdb-name.children remains unchanged
        ]

    # Default return if no conditions met
    return [
        dash.no_update,
        dash.no_update,
        "Upload a PDB file to visualize its Ramachandran plot and 3D structure.",
        dash.no_update,
        "Current PDB: 5xgu.pdb"
    ]

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
