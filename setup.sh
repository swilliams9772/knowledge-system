# Create virtual environment
conda create -n ksa python=3.9
conda activate ksa

# Install dependencies
pip install -e .

# Set up environment variables
export OPENAI_API_KEY=<YOUR_API_KEY>
export OCR_SERVER_ADDRESS=http://localhost:8000/ocr/

# Start services
python ksa/ocr_server.py &
python ksa/retrieval_server.py & 