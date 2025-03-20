"""
Entry point for Azure App Service.
Punto de entrada para Azure App Services.
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Import the FastAPI application
from app.backend.api.main import app

# This is the object that Azure App Service looks for
# Este es el objeto que Azure App Services busca
app = app 