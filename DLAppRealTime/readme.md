### make .venv:
- py -3.11 -m venv .venv
- .venv/scripts/activate

### Requirements
ffmpeg should be installed in your PC.
official site: https://www.ffmpeg.org/download.html
descriptio: https://jp.videoproc.com/edit-convert/how-to-download-and-install-ffmpeg.htm

Use .venv with: 
- python version == 3.11.7
- pip install -r req.txt

If you got an error with installing torch+cpu, try this
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd pytorch/huggingface/audio_course