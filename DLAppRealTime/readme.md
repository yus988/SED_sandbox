make .venv:
- py -3.11 -m venv .venv
- .venv/scripts/activate

Use .venv with: 
- python version == 3.11.7
- pip install -r req.txt

If you got an error with installing torch+cpu, try this
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121