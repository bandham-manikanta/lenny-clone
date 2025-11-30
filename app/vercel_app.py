"""
Vercel-compatible entry point for Streamlit
"""

import os
import sys
from pathlib import Path

# Ensure the app directory is in the path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(app_dir.parent))

# Set Streamlit configuration
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'

# Import and run the main app
from main import main

if __name__ == "__main__":
    main()
