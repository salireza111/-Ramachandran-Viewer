import dash
from dash import dcc, html, Input, Output, State
import base64
import os
import io
import py3Dmol
import plotly.graph_objects as go
import numpy as np
from Bio import PDB
from Bio.PDB import PPBuilder, DSSP

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
        <style>
          thead th {
            position: sticky;
            top: 0;
            background: white;
            z-index: 2;
          }
        </style>
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

def find_residue_position(chain_id, res_id, all_residues):
    for i,res in enumerate(all_residues):
        if res['chain']==chain_id and res['index']==res_id:
            return i
    return None

def parse_pdb(pdb_content, pdb_filename=DEFAULT_PDB):
    if pdb_filename != DEFAULT_PDB:
        save_path = os.path.join(UPLOAD_DIRECTORY, pdb_filename)
        with open(save_path, 'w') as f:
            f.write(pdb_content)
        pdb_filepath = save_path
    else:
        if not os.path.isfile(DEFAULT_PDB):
            return [],[],[]
        pdb_filepath = DEFAULT_PDB

    parser=PDB.PDBParser(QUIET=True)
    try:
        structure=parser.get_structure('uploaded', pdb_filepath)
    except:
        return [],[],[]

    models=list(structure.get_models())
    if not models:
        return [],[],[]
    model=models[0]

    secondary_structure={}
    for line in pdb_content.split('\n'):
        if line.startswith('HELIX'):
            chain_id=line[19]
            try:
                start_res_num=int(line[21:25].strip())
                end_res_num=int(line[33:37].strip())
                for res_num in range(start_res_num, end_res_num+1):
                    secondary_structure[(chain_id,res_num)]='Helix'
            except ValueError:
                continue
        elif line.startswith('SHEET'):
            chain_id=line[21]
            try:
                start_res_num=int(line[22:26].strip())
                end_res_num=int(line[33:37].strip())
                for res_num in range(start_res_num,end_res_num+1):
                    secondary_structure[(chain_id,res_num)]='Sheet'
            except ValueError:
                continue

    ppb=PPBuilder()
    peptides=ppb.build_peptides(model)
    if not peptides:
        return [],[],[]

    phi_psi_data=[]
    all_residues=[]
    residue_info=[]

    
    for pp in peptides:
        for res in pp:
            chain_id=res.get_parent().get_id()
            res_id=res.get_id()[1]
            atom_coords={}
            for atom_name in ['N','CA','C','O']:
                if atom_name in res:
                    coord=res[atom_name].get_coord()
                    atom_coords[atom_name]=coord.tolist()
                else:
                    atom_coords[atom_name]=None
            all_residues.append({
                'chain':chain_id,
                'index':res_id,
                'atom_coords':atom_coords
            })

    # Extract phi/psi
    for pp in peptides:
        phi_psi_list=pp.get_phi_psi_list()
        for i,res in enumerate(pp):
            phi,psi=phi_psi_list[i]
            if None not in [phi,psi]:
                chain_id=res.get_parent().get_id()
                res_id=res.get_id()[1]
                res_name=res.get_resname()
                sec_struct=secondary_structure.get((chain_id,res_id),'C')
                pos=find_residue_position(chain_id,res_id,all_residues)
                coords=all_residues[pos]['atom_coords'] if pos is not None else {}
                phi_psi_data.append({
                    'residue':res_name,
                    'index':res_id,
                    'chain':chain_id,
                    'phi':phi,
                    'psi':psi,
                    'sec_struct':sec_struct,
                    'atom_coords': coords
                })
                residue_info.append(f"{res_name} {res_id} {sec_struct}")

    # Initialize DSSP fields
    # DSSP fields: (index, AA, SS, ACC, phi_d, psi_d, NH->O_1_relidx, NH->O_1_energy,
    # O->NH_1_relidx, O->NH_1_energy, NH->O_2_relidx, NH->O_2_energy, O->NH_2_relidx, O->NH_2_energy)
    for e in phi_psi_data:
        for col in ["AA","DSSP_STRUCT","ACC","PHI_DSSP","PSI_DSSP",
                    "NH_O_1_relidx","NH_O_1_energy","O_NH_1_relidx","O_NH_1_energy",
                    "NH_O_2_relidx","NH_O_2_energy","O_NH_2_relidx","O_NH_2_energy"]:
            e[col]=''

    
    try:
        dssp=DSSP(model, pdb_filepath)
        dssp_map={}
        for k in dssp.keys():
            chain_id=k[0]
            resseq=k[1][1]
            v=dssp[k]
            # Extract all DSSP fields
            (d_idx, AA, SS, ACC, phi_d, psi_d, nh_o1_r, nh_o1_e, o_nh1_r, o_nh1_e, nh_o2_r, nh_o2_e, o_nh2_r, o_nh2_e) = v
            DSSP_STRUCT = SS if SS!=' ' else ''
            dssp_map[(chain_id,resseq)] = (AA,DSSP_STRUCT,ACC,phi_d,psi_d,
                                           nh_o1_r, nh_o1_e, o_nh1_r, o_nh1_e,
                                           nh_o2_r, nh_o2_e, o_nh2_r, o_nh2_e)

        
        for e in phi_psi_data:
            k=(e['chain'],e['index'])
            if k in dssp_map:
                (AA,DSSP_STRUCT,ACC,phi_d,psi_d,nh_o1_r,nh_o1_e,o_nh1_r,o_nh1_e,
                 nh_o2_r,nh_o2_e,o_nh2_r,o_nh2_e)=dssp_map[k]
                e['AA']=AA
                e['DSSP_STRUCT']=DSSP_STRUCT
                e['ACC']=f"{ACC:.3f}"
                if phi_d is not None:
                    e['PHI_DSSP']=f"{phi_d:.1f}"
                if psi_d is not None:
                    e['PSI_DSSP']=f"{psi_d:.1f}"
                e['NH_O_1_relidx']=str(nh_o1_r)
                e['NH_O_1_energy']=f"{nh_o1_e:.1f}" if nh_o1_e is not None else ''
                e['O_NH_1_relidx']=str(o_nh1_r)
                e['O_NH_1_energy']=f"{o_nh1_e:.1f}" if o_nh1_e is not None else ''
                e['NH_O_2_relidx']=str(nh_o2_r)
                e['NH_O_2_energy']=f"{nh_o2_e:.1f}" if nh_o2_e is not None else ''
                e['O_NH_2_relidx']=str(o_nh2_r)
                e['O_NH_2_energy']=f"{o_nh2_e:.1f}" if o_nh2_e is not None else ''
    except:
        
        pass

    return phi_psi_data,all_residues,residue_info

def generate_3d_viewer(full_pdb_data, selected_residue=None, all_residues=None):
    viewer=py3Dmol.view(width=600,height=500)
    if full_pdb_data.strip():
        viewer.addModel(full_pdb_data,"pdb")
    else:
        return viewer._make_html()
    viewer.setStyle({'cartoon':{'color':'spectrum','opacity':0.5}})

    if selected_residue and all_residues:
        pos=find_residue_position(selected_residue['chain'],selected_residue['index'],all_residues)
        if pos is not None:
            viewer.setStyle(
                {'resi':str(selected_residue['index']),'chain':selected_residue['chain']},
                {'stick':{'color':'red','radius':0.3}}
            )
            if pos>0:
                prev_res=all_residues[pos-1]
                viewer.setStyle({'resi':str(prev_res['index']),'chain':prev_res['chain']},
                                {'stick':{'color':'blue','radius':0.2}})
            if pos<len(all_residues)-1:
                next_res=all_residues[pos+1]
                viewer.setStyle({'resi':str(next_res['index']),'chain':next_res['chain']},
                                {'stick':{'color':'green','radius':0.2}})
            for atom_name in ['N','CA','C']:
                coord=selected_residue['atom_coords'].get(atom_name)
                if coord:
                    viewer.addSphere({'center':{'x':coord[0],'y':coord[1],'z':coord[2]},
                                      'radius':0.2,'color':'yellow'})
                    viewer.addLabel(
                        atom_name,
                        {
                            'position':{'x':coord[0],'y':coord[1],'z':coord[2]},
                            'backgroundColor':'white',
                            'fontSize':10,
                            'fontColor':'black'
                        }
                    )
            phi=selected_residue['phi']
            psi=selected_residue['psi']
            if pos>0:
                prev_res=all_residues[pos-1]
                c_prev=prev_res['atom_coords'].get('C')
                n=selected_residue['atom_coords'].get('N')
                ca=selected_residue['atom_coords'].get('CA')
                c=selected_residue['atom_coords'].get('C')
                if all([c_prev,n,ca,c]):
                    viewer.addCylinder({
                        'start':{'x':c_prev[0],'y':c_prev[1],'z':c_prev[2]},
                        'end':{'x':c[0],'y':c[1],'z':c[2]},
                        'color':'green','radius':0.1
                    })
                    mid_phi=[(c_prev[0]+c[0])/2,(c_prev[1]+c[1])/2,(c_prev[2]+c[2])/2]
                    viewer.addLabel(f"Phi: {np.degrees(phi):.1f}°",{
                        'position':{'x':mid_phi[0],'y':mid_phi[1],'z':mid_phi[2]},
                        'backgroundColor':'white','fontSize':10,'fontColor':'black'
                    })
            if pos<len(all_residues)-1:
                next_res=all_residues[pos+1]
                n=selected_residue['atom_coords'].get('N')
                ca=selected_residue['atom_coords'].get('CA')
                c=selected_residue['atom_coords'].get('C')
                n_next=next_res['atom_coords'].get('N')
                if all([n,ca,c,n_next]):
                    viewer.addCylinder({
                        'start':{'x':n[0],'y':n[1],'z':n[2]},
                        'end':{'x':n_next[0],'y':n_next[1],'z':n_next[2]},
                        'color':'orange','radius':0.1
                    })
                    mid_psi=[(n[0]+n_next[0])/2,(n[1]+n_next[1])/2,(n[2]+n_next[2])/2]
                    viewer.addLabel(
                        f"Psi: {np.degrees(psi):.1f}°",
                        {
                            'position':{'x':mid_psi[0],'y':mid_psi[1],'z':mid_psi[2]},
                            'backgroundColor':'white',
                            'fontSize':10,
                            'fontColor':'black'
                        }
                    )
            viewer.zoomTo({'resi':str(selected_residue['index']),'chain':selected_residue['chain']})
    viewer.render()
    return viewer._make_html()

def create_ramachandran_plot(phi_psi_data,residue_info):
    phi_psi=np.array([[item['phi'],item['psi']] for item in phi_psi_data]) if phi_psi_data else np.empty((0,2))
    secondary_structure_colors={'Helix':'blue','Sheet':'yellow','C':'grey'}

    if phi_psi_data:
        x_values=phi_psi[:,0]*180/np.pi
        y_values=phi_psi[:,1]*180/np.pi
        marker_colors=[secondary_structure_colors.get(item['sec_struct'],'grey') for item in phi_psi_data]
    else:
        x_values,y_values,marker_colors=[],[],[]

    fig=go.Figure([go.Scatter(
        x=x_values,y=y_values,
        mode='markers',
        marker=dict(color=marker_colors,size=6,opacity=0.7),
        text=residue_info,
        customdata=[{'index':item['index'],'chain':item['chain']} for item in phi_psi_data],
        hoverinfo='text'
    )])

    for sec_struct,color in secondary_structure_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],y=[None],mode='markers',
            marker=dict(size=10,color=color),
            name=sec_struct
        ))

    fig.update_layout(
        title='Ramachandran Plot',
        width=600,height=500,
        xaxis_title='Phi (degrees)',
        yaxis_title='Psi (degrees)',
        showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
        xaxis=dict(range=[-180,180]),
        yaxis=dict(range=[-180,180]),
        template='none',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black')
    )
    return fig

phi_psi_data_default, all_residues_default, residue_info_default=[],[],[]
if os.path.isfile(DEFAULT_PDB):
    with open(DEFAULT_PDB) as f:
        default_pdb_data=f.read()
    phi_psi_data_default,all_residues_default,residue_info_default=parse_pdb(default_pdb_data,DEFAULT_PDB)
    viewer_html_default=generate_3d_viewer(default_pdb_data,None,all_residues_default)
    ramachandran_fig_default=create_ramachandran_plot(phi_psi_data_default,residue_info_default)
else:
    default_pdb_data=""
    viewer_html_default=""
    ramachandran_fig_default=create_ramachandran_plot([],[])

app.layout=html.Div([
    html.Header([
        html.H1("Molecular Visualization Dashboard",className="text-4xl font-bold text-center text-white mb-4"),
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
                html.A('Select a PDB File',className="text-blue-500 underline hover:text-blue-700 transition-colors")
            ]),
            className="border-2 border-dashed border-blue-400 bg-gray-50 p-8 rounded-lg text-center cursor-pointer hover:bg-blue-100 transition-colors",
            multiple=False, accept='.pdb'
        ),
    ], className="max-w-xl mx-auto mb-8"),

    html.Div(id='pdb-name',className="text-center text-2xl font-semibold text-gray-800 mb-6"),

    html.Div([
        html.Div([
            dcc.Loading(
                id="loading-plot",
                type="default",
                children=[
                    dcc.Graph(
                        id='ramachandran-plot',
                        figure=ramachandran_fig_default,
                        config={'displayModeBar':False},
                        style={'width':'600px','height':'500px'}
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
                                style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
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
        html.Div(id='residue-table-container',
                 className="bg-white rounded-lg shadow-md text-gray-800 text-sm hover:shadow-lg transition-shadow overflow-y-auto max-h-72 p-4",
                 style={'width':'1280px'})
    ], className="max-w-10xl mx-auto flex justify-center"),

    dcc.Location(id='scroll-loc',refresh=False),
    dcc.Store(id='pdb-data',storage_type='memory')
], className="py-8 space-y-8")

@app.callback(
    [
        Output('ramachandran-plot','figure'),
        Output('3d-viewer','children'),
        Output('pdb-data','data'),
        Output('pdb-name','children'),
        Output('residue-table-container','children'),
        Output('scroll-loc','href')
    ],
    [Input('upload-data','contents'),
     Input('ramachandran-plot','clickData')],
    [State('upload-data','filename'),
     State('pdb-data','data')]
)
def update_visualizations(upload_contents, clickData, upload_filename, cached_data):
    def build_residue_table(phi_psi_data_current, selected_res_index=None):
        if not phi_psi_data_current:
            return "No residues to display."
        column_widths = {
            "Residue": "150px",
            "Index": "100px",
            "Phi": "120px",
            "Psi": "120px",
            "Sec.Struct": "200px",
            
    }

        table_header=html.Thead(
            html.Tr([
                html.Th("Residue"),html.Th("Index"),html.Th("Phi"),html.Th("Psi"),html.Th("Sec.Struct"),
                html.Th("AA"),html.Th("DSSP_STRUCT"),html.Th("ACC"),html.Th("PHI_DSSP"),html.Th("PSI_DSSP"),
                html.Th("NH->O_1_rel"),html.Th("NH->O_1_eng"),html.Th("O->NH_1_rel"),html.Th("O->NH_1_eng"),
                html.Th("NH->O_2_rel"),html.Th("NH->O_2_eng"),html.Th("O->NH_2_rel"),html.Th("O->NH_2_eng")
            ], className="text-center")
        )
        rows=[]
        for item in phi_psi_data_current:
            tr_classes="hover:bg-gray-100"
            if selected_res_index is not None and item['index']==selected_res_index:
                tr_classes+=" bg-yellow-200"
            rows.append(
                html.Tr([
                    html.Td(item['residue'],className="text-center"),
                    html.Td(str(item['index']),className="text-center"),
                    html.Td(f"{np.degrees(item['phi']):.1f}°",className="text-center"),
                    html.Td(f"{np.degrees(item['psi']):.1f}°",className="text-center"),
                    html.Td(item['sec_struct'],className="text-center"),
                    html.Td(item.get('AA',''),className="text-center"),
                    html.Td(item.get('DSSP_STRUCT',''),className="text-center"),
                    html.Td(item.get('ACC',''),className="text-center"),
                    html.Td(item.get('PHI_DSSP',''),className="text-center"),
                    html.Td(item.get('PSI_DSSP',''),className="text-center"),
                    html.Td(item.get('NH_O_1_relidx',''),className="text-center"),
                    html.Td(item.get('NH_O_1_energy',''),className="text-center"),
                    html.Td(item.get('O_NH_1_relidx',''),className="text-center"),
                    html.Td(item.get('O_NH_1_energy',''),className="text-center"),
                    html.Td(item.get('NH_O_2_relidx',''),className="text-center"),
                    html.Td(item.get('NH_O_2_energy',''),className="text-center"),
                    html.Td(item.get('O_NH_2_relidx',''),className="text-center"),
                    html.Td(item.get('O_NH_2_energy',''),className="text-center")
                ], className=tr_classes)
            )
        return html.Table([table_header,html.Tbody(rows)],className="table-auto border-collapse w-full whitespace-nowrap")

    ctx=dash.callback_context

    if not os.path.isfile(DEFAULT_PDB):
        
        return create_ramachandran_plot([],[]),html.Iframe('',style={'width':'600px','height':'500px'}),dash.no_update,"No Default PDB","No residues",dash.no_update

    if not ctx.triggered:
        
        return (
            create_ramachandran_plot(phi_psi_data_default,residue_info_default),
            html.Iframe(
                srcDoc=viewer_html_default,
                style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
                className="rounded-lg shadow-inner"
            ),
            dash.no_update,
            "Current PDB: 5xgu.pdb",
            build_residue_table(phi_psi_data_default),
            dash.no_update
        )

    triggered_id=ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id=='upload-data' and upload_contents is not None:
        try:
            content_type, content_string=upload_contents.split(',')
            decoded=base64.b64decode(content_string)
            pdb_content=decoded.decode('utf-8')
        except:
            return (
                create_ramachandran_plot(phi_psi_data_default,residue_info_default),
                html.Iframe(srcDoc=viewer_html_default,style={'width':'600px','height':'500px','border':'none','overflow':'hidden'}),
                dash.no_update,
                "Current PDB: 5xgu.pdb",
                build_residue_table(phi_psi_data_default),
                dash.no_update
            )
        phi_psi_data, all_residues, residue_info = parse_pdb(pdb_content, upload_filename)
        fig=create_ramachandran_plot(phi_psi_data,residue_info)
        vhtml=generate_3d_viewer(pdb_content,None,all_residues)
        pdb_store_data={
            'pdb_content':pdb_content,
            'phi_psi_data':phi_psi_data,
            'all_residues':all_residues,
            'residue_info':residue_info
        }
        pdb_name=f"Current PDB: {upload_filename}"
        table_component=build_residue_table(phi_psi_data)
        return (
            fig,
            html.Iframe(
                srcDoc=vhtml,
                style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
                className="rounded-lg shadow-inner"
            ),
            pdb_store_data,
            pdb_name,
            table_component,
            dash.no_update
        )

    elif triggered_id=='ramachandran-plot' and clickData is not None:
        if cached_data and 'pdb_content' in cached_data:
            pdb_content=cached_data['pdb_content']
            phi_psi_data_current=cached_data['phi_psi_data']
            all_residues_current=cached_data['all_residues']
            residue_info_current=cached_data['residue_info']
        else:
            with open(DEFAULT_PDB) as f:
                def_pdb_data=f.read()
            phi_psi_data_current,all_residues_current,residue_info_current=parse_pdb(def_pdb_data,DEFAULT_PDB)
            pdb_content=def_pdb_data
        try:
            residue_info_clicked=clickData['points'][0]['customdata']
            residue_index=residue_info_clicked['index']
            residue_chain=residue_info_clicked['chain']
        except:
            fig=create_ramachandran_plot(phi_psi_data_current,residue_info_current)
            table_component=build_residue_table(phi_psi_data_current)
            return (
                fig,
                html.Iframe(
                    srcDoc=generate_3d_viewer(pdb_content,None,all_residues_current),
                    style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
                    className="rounded-lg shadow-inner"
                ),
                dash.no_update,
                "Current PDB: 5xgu.pdb",
                table_component,
                dash.no_update
            )

        selected_residue=next((item for item in phi_psi_data_current if item['index']==residue_index and item['chain']==residue_chain),None)
        fig=create_ramachandran_plot(phi_psi_data_current,residue_info_current)
        if selected_residue is None:
            table_component=build_residue_table(phi_psi_data_current)
            return (
                fig,
                html.Iframe(
                    srcDoc=generate_3d_viewer(pdb_content,None,all_residues_current),
                    style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
                    className="rounded-lg shadow-inner"
                ),
                dash.no_update,
                "Current PDB: 5xgu.pdb",
                table_component,
                dash.no_update
            )
        vhtml=generate_3d_viewer(pdb_content,selected_residue,all_residues_current)
        table_component=build_residue_table(phi_psi_data_current, selected_res_index=selected_residue['index'])
        pname="Current PDB: "+(upload_filename if cached_data and 'pdb_content' in cached_data and upload_filename else "5xgu.pdb")
        return (
            fig,
            html.Iframe(
                srcDoc=vhtml,
                style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
                className="rounded-lg shadow-inner"
            ),
            dash.no_update,
            pname,
            table_component,
            dash.no_update
        )

    
    fig=create_ramachandran_plot(phi_psi_data_default,residue_info_default)
    table_component="No residues"
    if phi_psi_data_default:
        table_component=build_residue_table(phi_psi_data_default)
    return (
        fig,
        html.Iframe(
            srcDoc=viewer_html_default,
            style={'width':'600px','height':'500px','border':'none','overflow':'hidden'},
            className="rounded-lg shadow-inner"
        ),
        dash.no_update,
        "Current PDB: 5xgu.pdb",
        table_component,
        dash.no_update
    )

if __name__=='__main__':
    app.run_server(debug=True)
