#Final without PDB Fetch

import dash
from dash import dcc, html, Input, Output, State
import base64
import io
import py3Dmol
import plotly.graph_objects as go
import numpy as np
from Bio import PDB
from Bio.PDB import PPBuilder

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the Flask server for deployments

# Default PDB file path
DEFAULT_PDB = '5xgu.pdb'

# Load the default PDB file
with open(DEFAULT_PDB, 'r') as f:
    default_pdb_data = f.read()

# Helper Functions

def parse_pdb(pdb_content):
    """
    Parses PDB data from a string and extracts phi/psi angles and secondary structures.
    
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
            all_residues.append({
                'chain': chain_id,
                'index': res_id,
                'residue_obj': res
            })
    
    for pp in ppb.build_peptides(structure):
        for i, res in enumerate(pp):
            phi, psi = pp.get_phi_psi_list()[i]
            if None not in [phi, psi]:
                res_name = res.get_resname()
                res_id = res.get_id()[1]
                chain_id = res.get_parent().get_id()
                sec_struct = secondary_structure.get((chain_id, res_id), 'Unknown')
                
                phi_psi_data.append({
                    'residue': res_name,
                    'index': res_id,
                    'chain': chain_id,
                    'phi': phi,
                    'psi': psi,
                    'sec_struct': sec_struct,
                    'residue_obj': res
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

def get_atom_coordinates(residue, atom_name):
    """
    Retrieves the coordinates of a specified atom in a residue.
    
    Parameters:
        residue (dict): Residue information dictionary.
        atom_name (str): Name of the atom (e.g., 'CA').
        
    Returns:
        tuple or None: (x, y, z) coordinates as floats if atom exists, else None.
    """
    try:
        atom = residue['residue_obj'][atom_name]
        coord = atom.get_coord()
        return float(coord[0]), float(coord[1]), float(coord[2])
    except KeyError:
        return None

def visualize_dihedral_angles(viewer, selected_residue, all_residues):
    """
    Visualizes Phi and Psi dihedral angles using cylinders and labels in the 3D viewer.
    
    Parameters:
        viewer (py3Dmol.view): The 3D viewer instance.
        selected_residue (dict): The selected residue information.
        all_residues (list): List of all residues.
        
    Returns:
        py3Dmol.view: Updated viewer with dihedral angles visualized.
    """
    pos = find_residue_position(selected_residue['chain'], selected_residue['index'], all_residues)
    if pos is None:
        return viewer
    
    phi = selected_residue['phi']
    psi = selected_residue['psi']
    
    # Phi: C(i-1) - N(i) - CA(i) - C(i)
    if pos > 0:
        prev_res = all_residues[pos - 1]
        c_prev = get_atom_coordinates(prev_res, 'C')
        n = get_atom_coordinates(selected_residue, 'N')
        ca = get_atom_coordinates(selected_residue, 'CA')
        c = get_atom_coordinates(selected_residue, 'C')
        if None not in [c_prev, n, ca, c]:
            # Draw cylinder for Phi
            viewer.addCylinder({
                'start': {'x': c_prev[0], 'y': c_prev[1], 'z': c_prev[2]},
                'end': {'x': c[0], 'y': c[1], 'z': c[2]},
                'color': 'green',
                'radius': 0.1
            })
            # Label for Phi
            mid_phi = [(c_prev[0] + c[0]) / 2, (c_prev[1] + c[1]) / 2, (c_prev[2] + c[2]) / 2]
            viewer.addLabel(
                f"Phi: {np.degrees(phi):.1f}°",
                {
                    'position': {'x': float(mid_phi[0]), 'y': float(mid_phi[1]), 'z': float(mid_phi[2])},
                    'backgroundColor': 'white',
                    'fontSize': 10,
                    'fontColor': 'black'
                }
            )
    
    # Psi: N(i) - CA(i) - C(i) - N(i+1)
    if pos < len(all_residues) - 1:
        next_res = all_residues[pos + 1]
        n = get_atom_coordinates(selected_residue, 'N')
        ca = get_atom_coordinates(selected_residue, 'CA')
        c = get_atom_coordinates(selected_residue, 'C')
        n_next = get_atom_coordinates(next_res, 'N')
        if None not in [n, ca, c, n_next]:
            # Draw cylinder for Psi
            viewer.addCylinder({
                'start': {'x': n[0], 'y': n[1], 'z': n[2]},
                'end': {'x': n_next[0], 'y': n_next[1], 'z': n_next[2]},
                'color': 'orange',
                'radius': 0.1
            })
            # Label for Psi
            mid_psi = [(n[0] + n_next[0]) / 2, (n[1] + n_next[1]) / 2, (n[2] + n_next[2]) / 2]
            viewer.addLabel(
                f"Psi: {np.degrees(psi):.1f}°",
                {
                    'position': {'x': float(mid_psi[0]), 'y': float(mid_psi[1]), 'z': float(mid_psi[2])},
                    'backgroundColor': 'white',
                    'fontSize': 10,
                    'fontColor': 'black'
                }
            )
    
    return viewer

def generate_3d_viewer(full_pdb_data, selected_residue=None, all_residues=None):
    """
    Generates the HTML for the 3D molecular viewer with optional atom annotations.
    
    Parameters:
        full_pdb_data (str): The full PDB data as a string.
        selected_residue (dict): The selected residue data.
        all_residues (list): List of all residues.
        
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
            
            # Add label to selected residue
            ca_coords = get_atom_coordinates(selected_residue, 'CA')
            if ca_coords:
                viewer.addLabel(
                    f"{selected_residue['residue']} {selected_residue['index']}",
                    {
                        'position': {'x': float(ca_coords[0]), 'y': float(ca_coords[1]), 'z': float(ca_coords[2])},
                        'backgroundColor': 'white',
                        'fontSize': 12,
                        'fontColor': 'black'
                    }
                )
            
            # Visualize dihedral angles
            viewer = visualize_dihedral_angles(viewer, selected_residue, all_residues)
            
            # Annotate atoms within the selected residue
            for atom in selected_residue['residue_obj']:
                atom_name = atom.get_name()
                atom_coords = atom.get_coord()
                viewer.addLabel(
                    atom_name,
                    {
                        'position': {
                            'x': float(atom_coords[0]),
                            'y': float(atom_coords[1]),
                            'z': float(atom_coords[2])
                        },
                        'backgroundColor': 'yellow',
                        'fontSize': 8,
                        'fontColor': 'black',
                        'fontWeight': 'bold'
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
            size=6
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
    
    # Upload Button
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
    
    html.Div([
        # Ramachandran Plot
        html.Div([
            dcc.Graph(
                id='ramachandran-plot',
                figure=ramachandran_fig_default
            )
        ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # 3D Viewer
        html.Div([
            html.Div(id='3d-viewer', children=[
                html.Iframe(srcDoc=viewer_html_default, width='800px', height='600px')
            ])
        ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Residue Information Panel
    html.Div(id='residue-info', style={'marginTop': '20px', 'fontSize': '18px', 'textAlign': 'center'})
], style={'width': '90%', 'margin': '0 auto'})

# Combined Callback to Handle Uploads and Clicks
@app.callback(
    [
        Output('ramachandran-plot', 'figure'),
        Output('3d-viewer', 'children'),
        Output('residue-info', 'children')
    ],
    [
        Input('upload-data', 'contents'),
        Input('ramachandran-plot', 'clickData')
    ],
    [
        State('upload-data', 'filename')
    ]
)
def update_visualizations(upload_contents, clickData, upload_filename):
    ctx = dash.callback_context

    if not ctx.triggered:
        # No action needed if no triggers
        return dash.no_update, dash.no_update, "Upload a PDB file to visualize its Ramachandran plot and 3D structure."

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'upload-data' and upload_contents is not None:
        # Handle file upload
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            pdb_content = decoded.decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return dash.no_update, dash.no_update, "Error: The uploaded file is not a valid PDB text file."

        # Parse the uploaded PDB content
        phi_psi_data, all_residues, residue_info = parse_pdb(pdb_content)

        if not phi_psi_data:
            return dash.no_update, dash.no_update, "No valid residues with phi/psi angles found in the uploaded file."

        # Create Ramachandran Plot
        fig = create_ramachandran_plot(phi_psi_data, residue_info)

        # Generate 3D Viewer HTML
        viewer_html = generate_3d_viewer(full_pdb_data=pdb_content, selected_residue=None, all_residues=all_residues)

        return fig, html.Div([
            html.Iframe(srcDoc=viewer_html, width='800px', height='600px')
        ]), "Upload successful. Please select a residue in the Ramachandran plot to view details."

    elif triggered_id == 'ramachandran-plot' and clickData is not None:
        # Handle click on Ramachandran plot
        try:
            residue_info = clickData['points'][0]['customdata']
            residue_index = residue_info['index']
            residue_chain = residue_info['chain']
        except (IndexError, KeyError, TypeError):
            return dash.no_update, dash.no_update, "Error retrieving residue information from the plot."

        # Determine the current PDB data being viewed
        if upload_contents is not None:
            try:
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                pdb_content = decoded.decode('utf-8')
            except (ValueError, UnicodeDecodeError):
                return dash.no_update, dash.no_update, "Error: The uploaded file is not a valid PDB text file."
        else:
            pdb_content = default_pdb_data
            phi_psi_data_current = phi_psi_data_default
            all_residues_current = all_residues_default
            residue_info_current = residue_info_default

        # Parse PDB content to get residues
        if upload_contents is not None:
            phi_psi_data_current, all_residues_current, residue_info_current = parse_pdb(pdb_content)
        else:
            phi_psi_data_current = phi_psi_data_default
            all_residues_current = all_residues_default
            residue_info_current = residue_info_default

        # Find the selected residue
        selected_residue = next((item for item in phi_psi_data_current if item['index'] == residue_index and item['chain'] == residue_chain), None)

        if selected_residue is None:
            return dash.no_update, dash.no_update, "Selected residue not found."

        # Generate updated 3D viewer with highlighted residues and atom annotations
        viewer_html = generate_3d_viewer(full_pdb_data=pdb_content, selected_residue=selected_residue, all_residues=all_residues_current)

        # Prepare residue information text
        residue_text = (
            f"**Residue:** {selected_residue['residue']} {selected_residue['index']}<br>"
            f"**Chain:** {selected_residue['chain']}<br>"
            f"**Secondary Structure:** {selected_residue['sec_struct']}<br>"
            f"**Phi Angle:** {np.degrees(selected_residue['phi']):.1f}°<br>"
            f"**Psi Angle:** {np.degrees(selected_residue['psi']):.1f}°"
        )

        return (
            dash.no_update,  # Ramachandran plot remains unchanged
            html.Div([
                html.Iframe(srcDoc=viewer_html, width='800px', height='600px')
            ]),
            html.Div([
                html.H3("Selected Residue Information"),
                html.P(dcc.Markdown(residue_text))
            ])
        )

    # Default return if no conditions met
    return dash.no_update, dash.no_update, "Upload a PDB file to visualize its Ramachandran plot and 3D structure."

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
