# Ramachandran Viewer
---
##### Dec 09, 2024 | S.Alireza Hashemi



<div style="text-align: justify"> 
Dash-based interactive and dynamic Ramachandran plot viewer with the ability to map the plot on a 3d representation of a PDB file.

* Denoted structures extracted from PDB file, de novo models could act tricky due to the lack of the proper annotation; this could be fixed through the DSSP implementation.
</div>

![Ramachandran](./screenshot.png)

 
Installation guide:

```bash
# Git clone
git clone https://github.com/salireza111/-Ramachandran-Viewer.git

# Create the virtual environment
python -m venv rama-env

# Activate the environment
source rama-env/bin/activate  # On macOS/Linux
rama-env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the code
python main.py
```

and it will be accessible via http://127.0.0.1:8050/ on your browser.