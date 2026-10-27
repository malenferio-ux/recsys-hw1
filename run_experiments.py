"""Execute the notebook with the current Python; save outputs even on failure."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    (root / '.cache').mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(root / '.cache' / 'matplotlib'))
    os.environ.setdefault('IPYTHONDIR', str(root / '.cache' / 'ipython'))
    path = root / 'experiments.ipynb'
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    def report(cell, cell_index, **kwargs):
        print(f'Cell {cell_index + 1}/{len(notebook.cells)} completed', flush=True)
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='', flush=True)
        nbformat.write(notebook, path)
    client.on_cell_executed = report
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print(f'Executed: {path}')


if __name__ == '__main__':
    main()
