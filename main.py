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

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 text-gray-900">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

UPLOAD_DIRECTORY = "uploaded_pdbs"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

DEFAULT_PDB = '5xgu.pdb'
try:
    with open(DEFAULT_PDB, 'r') as f:
        default_pdb_data = f.read()
except FileNotFoundError:
    default_pdb_data = ""
    print("Warning: 5xgu.pdb not found.")

def find_residue_position(chain_id, res_id, all_residues):
    for i, res in enumerate(all_residues):
        if res['chain'] == chain_id and res['index'] == res_id:
            return i
    return None

def parse_pdb(pdb_content):
    if not pdb_content.strip():
        return [], [], []
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('uploaded', io.StringIO(pdb_content))
    except Exception:
        return [], [], []

    models = list(structure.get_models())
    if not models:
        return [], [], []

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
                continue
        elif line.startswith('SHEET'):
            chain_id = line[21]
            try:
                start_res_num = int(line[22:26].strip())
                end_res_num = int(line[33:37].strip())
                for res_num in range(start_res_num, end_res_num + 1):
                    secondary_structure[(chain_id, res_num)] = 'Sheet'
            except ValueError:
                continue

    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure)
    if not peptides:
        return [], [], []

    phi_psi_data = []
    all_residues = []
    residue_info = []

    for pp in peptides:
        for res in pp:
            chain_id = res.get_parent().get_id()
            res_id = res.get_id()[1]
            atom_coords = {}
            for atom_name in ['N', 'CA', 'C', 'O']:
                if atom_name in res:
                    coord = res[atom_name].get_coord()
                    atom_coords[atom_name] = coord.tolist()
                else:
                    atom_coords[atom_name] = None
            all_residues.append({
                'chain': chain_id,
                'index': res_id,
                'atom_coords': atom_coords
            })

    for pp in peptides:
        phi_psi_list = pp.get_phi_psi_list()
        for i, res in enumerate(pp):
            phi, psi = phi_psi_list[i]
            if None not in [phi, psi]:
                res_name = res.get_resname()
                res_id = res.get_id()[1]
                chain_id = res.get_parent().get_id()
                sec_struct = secondary_structure.get((chain_id, res_id), 'Unknown')

                atom_coords = {}
                for atom_name in ['N', 'CA', 'C']:
                    if atom_name in res:
                        coord = res[atom_name].get_coord()
                        atom_coords[atom_name] = coord.tolist()
                    else:
                        atom_coords[atom_name] = None

                phi_psi_data.append({
                    'residue': res_name,
                    'index': res_id,
                    'chain': chain_id,
                    'phi': phi,
                    'psi': psi,
                    'sec_struct': sec_struct,
                    'atom_coords': atom_coords
                })
                residue_info.append(f"{res_name} {res_id} {sec_struct}")

    return phi_psi_data, all_residues, residue_info

def generate_3d_viewer(full_pdb_data, selected_residue=None, all_residues=None):
    viewer = py3Dmol.view(width=600, height=500)
    viewer.addModel(full_pdb_data, "pdb")
    
    viewer.setStyle({'cartoon': {'color': 'spectrum', 'opacity': 0.5}})

    if selected_residue and all_residues:
        pos = find_residue_position(selected_residue['chain'], selected_residue['index'], all_residues)
        if pos is not None:
            # main residue
            viewer.setStyle(
                {'resi': str(selected_residue['index']), 'chain': selected_residue['chain']},
                {'stick': {'color': 'red', 'radius': 0.3}}
            )

            # previous residue
            if pos > 0:
                prev_res = all_residues[pos - 1]
                viewer.setStyle(
                    {'resi': str(prev_res['index']), 'chain': prev_res['chain']},
                    {'stick': {'color': 'blue', 'radius': 0.2}}
                )

            # next residue
            if pos < len(all_residues) - 1:
                next_res = all_residues[pos + 1]
                viewer.setStyle(
                    {'resi': str(next_res['index']), 'chain': next_res['chain']},
                    {'stick': {'color': 'green', 'radius': 0.2}}
                )

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

            phi = selected_residue['phi']
            psi = selected_residue['psi']
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
                    mid_phi = [(c_prev[0] + c[0]) / 2, (c_prev[1] + c[1]) / 2, (c_prev[2] + c[2]) / 2]
                    viewer.addLabel(
                        f"Phi: {np.degrees(phi):.1f}°",
                        {
                            'position': {'x': mid_phi[0], 'y': mid_phi[1], 'z': mid_phi[2]},
                            'backgroundColor': 'white',
                            'fontSize': 10,
                            'fontColor': 'black'
                        }
                    )

            if pos < len(all_residues) - 1:
                next_res = all_residues[pos + 1]
                n = selected_residue['atom_coords'].get('N')
                ca = selected_residue['atom_coords'].get('CA')
                c = selected_residue['atom_coords'].get('C')
                n_next = next_res['atom_coords'].get('N')
                if all([n, ca, c, n_next]):
                    viewer.addCylinder({
                        'start': {'x': n[0], 'y': n[1], 'z': n[2]},
                        'end': {'x': n_next[0], 'y': n_next[1], 'z': n_next[2]},
                        'color': 'orange',
                        'radius': 0.1
                    })
                    mid_psi = [(n[0] + n_next[0]) / 2, (n[1] + n_next[1]) / 2, (n[2] + n_next[2]) / 2]
                    viewer.addLabel(
                        f"Psi: {np.degrees(psi):.1f}°",
                        {
                            'position': {'x': mid_psi[0], 'y': mid_psi[1], 'z': mid_psi[2]},
                            'backgroundColor': 'white',
                            'fontSize': 10,
                            'fontColor': 'black'
                        }
                    )

            viewer.zoomTo({'resi': str(selected_residue['index']), 'chain': selected_residue['chain']})

    viewer.render()
    return viewer._make_html()

def create_ramachandran_plot(phi_psi_data, residue_info):
    phi_psi = np.array([[item['phi'], item['psi']] for item in phi_psi_data])
    secondary_structure_colors = {'Helix': 'blue', 'Sheet': 'yellow', 'Unknown': 'grey'}

    fig = go.Figure()
    if len(phi_psi_data) > 0:
        x_values = phi_psi[:,0]*180/np.pi
        y_values = phi_psi[:,1]*180/np.pi
        marker_colors = [secondary_structure_colors.get(item['sec_struct'], 'grey') for item in phi_psi_data]
    else:
        x_values, y_values, marker_colors = [], [], []

    fig.add_trace(go.Scatter(
        x=x_values, y=y_values,
        mode='markers',
        marker=dict(color=marker_colors, size=6, opacity=0.7),
        text=residue_info,
        customdata=[{'index': item['index'], 'chain': item['chain']} for item in phi_psi_data],
        hoverinfo='text'
    ))

    for sec_struct, color in secondary_structure_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            name=sec_struct
        ))

    fig.update_layout(
        title='Ramachandran Plot',
        width=600,
        height=500,
        xaxis_title='Phi (degrees)',
        yaxis_title='Psi (degrees)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[-180,180]),
        yaxis=dict(range=[-180,180]),
        template='none',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black')
    )
    return fig

phi_psi_data_default, all_residues_default, residue_info_default = parse_pdb(default_pdb_data)
ramachandran_fig_default = create_ramachandran_plot(phi_psi_data_default, residue_info_default)
viewer_html_default = generate_3d_viewer(full_pdb_data=default_pdb_data, selected_residue=None, all_residues=all_residues_default)

app.layout = html.Div([
    html.Header([
        html.H1("Molecular Visualization Dashboard", className="text-4xl font-bold text-center text-white mb-4"),
        html.H2([
            "By: S.Alireza Hashemi ",
            html.A("salireza111.github.io", href="https://salireza111.github.io/", target="_blank",
                   className="text-yellow-200 underline hover:text-yellow-100 transition-colors")
        ], className="text-center text-lg text-yellow-100 mb-8")
    ], className="py-8 bg-gradient-to-r from-blue-600 to-indigo-600 shadow-md"),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a PDB File', className="text-blue-500 underline hover:text-blue-700 transition-colors")
            ]),
            className="border-2 border-dashed border-blue-400 bg-gray-50 p-8 rounded-lg text-center cursor-pointer hover:bg-blue-100 transition-colors",
            multiple=False,
            accept='.pdb'
        ),
    ], className="max-w-xl mx-auto mb-8"),

    html.Div(id='pdb-name', className="text-center text-2xl font-semibold text-gray-800 mb-6"),

    html.Div([
        html.Div([
            dcc.Loading(
                id="loading-plot",
                type="default",
                children=[
                    dcc.Graph(
                        id='ramachandran-plot',
                        figure=ramachandran_fig_default,
                        config={'displayModeBar': False},
                        style={'width': '600px', 'height': '500px'}
                    )
                ]
            )
        ], className="bg-white p-4 rounded-lg shadow-md flex justify-center items-center"),

        html.Div([
            dcc.Loading(
                id="loading-viewer",
                type="default",
                children=[
                    html.Div(
                        id='3d-viewer',
                        children=[
                            html.Iframe(
                                srcDoc=viewer_html_default,
                                style={'width': '600px', 'height': '500px', 'border': 'none', 'overflow': 'hidden'},
                                className="rounded-lg shadow-inner"
                            )
                        ],
                        className="flex justify-center items-center"
                    )
                ]
            )
        ], className="bg-white p-4 rounded-lg shadow-md flex justify-center items-center")
    ], className="flex flex-col md:flex-row justify-center items-center mb-8 space-y-6 md:space-y-0 md:space-x-4 px-4"),

    html.Div([
        html.Div(id='residue-info',
                 className="bg-white rounded-lg shadow-md text-gray-800 text-base hover:shadow-lg transition-shadow overflow-y-auto max-h-72 p-4",
                 style={'width':'635px'}),  
        html.Div(id='residue-table-container',
                 className="bg-white rounded-lg shadow-md text-gray-800 text-sm hover:shadow-lg transition-shadow overflow-y-auto max-h-72 p-4",
                 style={'width':'635px'})  
    ], className="max-w-10xl mx-auto flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4 justify-center"),

    dcc.Store(id='pdb-data', storage_type='memory')
], className="py-8 space-y-8")


@app.callback(
    [
        Output('ramachandran-plot', 'figure'),
        Output('3d-viewer', 'children'),
        Output('residue-info', 'children'),
        Output('pdb-data', 'data'),
        Output('pdb-name', 'children'),
        Output('residue-table-container', 'children')
    ],
    [Input('upload-data', 'contents'), Input('ramachandran-plot', 'clickData')],
    [State('upload-data', 'filename'), State('pdb-data', 'data')]
)
def update_visualizations(upload_contents, clickData, upload_filename, cached_data):
    def dssp_from_sec_struct(sec_struct):
        if sec_struct == 'Helix':
            return 'H'
        elif sec_struct == 'Sheet':
            return 'E'
        else:
            return 'C'

    def build_residue_table(phi_psi_data_current):
        if not phi_psi_data_current:
            return "No residues to display."
        table_header = html.Thead(html.Tr([
            html.Th("Residue", className="px-2 py-1 text-center"),
            html.Th("Index", className="px-2 py-1 text-center"),
            html.Th("Phi", className="px-2 py-1 text-center"),
            html.Th("Psi", className="px-2 py-1 text-center"),
            html.Th("Sec.Struct", className="px-2 py-1 text-center"),
            html.Th("DSSP", className="px-2 py-1 text-center")
        ]))
        rows = []
        for item in phi_psi_data_current:
            dssp_code = dssp_from_sec_struct(item['sec_struct'])
            rows.append(html.Tr([
                html.Td(item['residue'], className="px-2 py-1 text-center"),
                html.Td(str(item['index']), className="px-2 py-1 text-center"),
                html.Td(f"{np.degrees(item['phi']):.1f}°", className="px-2 py-1 text-center"),
                html.Td(f"{np.degrees(item['psi']):.1f}°", className="px-2 py-1 text-center"),
                html.Td(item['sec_struct'], className="px-2 py-1 text-center"),
                html.Td(dssp_code, className="px-2 py-1 text-center")
            ]))
        return html.Table([table_header, html.Tbody(rows)], className="table-auto border-collapse w-full whitespace-nowrap")

    ctx = dash.callback_context

    if not ctx.triggered:
        default_table = build_residue_table(phi_psi_data_default)
        return (
            dash.no_update,
            html.Iframe(
                srcDoc=viewer_html_default,
                style={'width': '600px', 'height': '500px', 'border': 'none', 'overflow': 'hidden'},
                className="rounded-lg shadow-inner"
            ),
            "Upload a PDB file to visualize its Ramachandran plot and 3D structure.",
            dash.no_update,
            "Current PDB: 5xgu.pdb",
            default_table
        )

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'upload-data' and upload_contents is not None:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            pdb_content = decoded.decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            default_table = build_residue_table(phi_psi_data_default)
            return dash.no_update, dash.no_update, "Error: Invalid PDB file.", dash.no_update, dash.no_update, default_table

        phi_psi_data, all_residues, residue_info = parse_pdb(pdb_content)
        if not phi_psi_data:
            empty_table = build_residue_table([])
            return dash.no_update, dash.no_update, "No valid residues found.", dash.no_update, dash.no_update, empty_table

        fig = create_ramachandran_plot(phi_psi_data, residue_info)
        viewer_html = generate_3d_viewer(full_pdb_data=pdb_content, selected_residue=None, all_residues=all_residues)
        pdb_store_data = {
            'pdb_content': pdb_content,
            'phi_psi_data': phi_psi_data,
            'all_residues': all_residues,
            'residue_info': residue_info
        }
        pdb_name = f"Current PDB: {upload_filename}"

        viewer_component = html.Iframe(
            srcDoc=viewer_html,
            style={'width': '600px', 'height': '500px', 'border': 'none', 'overflow': 'hidden'},
            className="rounded-lg shadow-inner"
        )

        residue_table_component = build_residue_table(phi_psi_data)

        return fig, viewer_component, "Upload successful. Select a residue in the Ramachandran plot.", pdb_store_data, pdb_name, residue_table_component

    elif triggered_id == 'ramachandran-plot' and clickData is not None:
        try:
            residue_info_clicked = clickData['points'][0]['customdata']
            residue_index = residue_info_clicked['index']
            residue_chain = residue_info_clicked['chain']
        except (IndexError, KeyError, TypeError):
            default_table = build_residue_table(phi_psi_data_default)
            return dash.no_update, dash.no_update, "Error retrieving residue info.", dash.no_update, dash.no_update, default_table

        if cached_data and 'pdb_content' in cached_data:
            pdb_content = cached_data['pdb_content']
            phi_psi_data_current = cached_data['phi_psi_data']
            all_residues_current = cached_data['all_residues']
            residue_info_current = cached_data['residue_info']
        else:
            pdb_content = default_pdb_data
            phi_psi_data_current = phi_psi_data_default
            all_residues_current = all_residues_default
            residue_info_current = residue_info_default

        selected_residue = next((item for item in phi_psi_data_current if item['index'] == residue_index and item['chain'] == residue_chain), None)
        if selected_residue is None:
            table_component = build_residue_table(phi_psi_data_current)
            return dash.no_update, dash.no_update, "Selected residue not found.", dash.no_update, dash.no_update, table_component

        viewer_html = generate_3d_viewer(full_pdb_data=pdb_content, selected_residue=selected_residue, all_residues=all_residues_current)
        residue_text = (
            f"**Residue:** {selected_residue['residue']} {selected_residue['index']}<br>"
            f"**Chain:** {selected_residue['chain']}<br>"
            f"**Secondary Structure:** {selected_residue['sec_struct']}<br>"
            f"**Phi Angle:** {np.degrees(selected_residue['phi']):.1f}°<br>"
            f"**Psi Angle:** {np.degrees(selected_residue['psi']):.1f}°"
        )

        viewer_component = html.Iframe(
            srcDoc=viewer_html,
            style={'width': '600px', 'height': '500px', 'border': 'none', 'overflow': 'hidden'},
            className="rounded-lg shadow-inner"
        )

        residue_info_component = html.Div([
            html.H3("Selected Residue Information", className="text-xl font-semibold mb-2"),
            html.P(dcc.Markdown(residue_text), className="text-gray-700")
        ], className="bg-white rounded-lg shadow-md text-gray-800 text-base hover:shadow-lg transition-shadow overflow-y-auto max-h-72 p-4",
        style={'width':'600px'})

        residue_table_component = build_residue_table(phi_psi_data_current)

        return (
            dash.no_update,
            viewer_component,
            residue_info_component,
            dash.no_update,
            dash.no_update,
            residue_table_component
        )

    default_table = build_residue_table(phi_psi_data_default)
    return dash.no_update, dash.no_update, "Upload a PDB file.", dash.no_update, "Current PDB: 5xgu.pdb", default_table

if __name__ == '__main__':
    app.run_server(debug=True)