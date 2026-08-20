"""Execute a notebook with this interpreter, without depending on a python alias."""
from pathlib import Path
import argparse
import sys
import nbformat
from nbclient import NotebookClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',nargs='?',default='05_opening_move_continuation.ipynb')
    args=parser.parse_args()
    path=Path(args.notebook).resolve()
    nb=nbformat.read(path,as_version=4)
    client=NotebookClient(nb,timeout=600,allow_errors=False,
                          resources={'metadata':{'path':str(path.parent)}})
    client.create_kernel_manager()
    client.km.kernel_spec.argv[0]=sys.executable
    client.execute()
    nbformat.write(nb,path)
    print(f'Executed {len(nb.cells)} cells successfully: {path}')


if __name__=='__main__':
    main()
