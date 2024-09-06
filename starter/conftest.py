import sys
import os

# Añade el directorio raíz del proyecto a sys.path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
