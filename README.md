# FaceID
[Thesis] Detection of Facial Deepfakes Using Occlusion and Head Rotation in Remote Identity Verification  

### Setup commands:

#### Powershell:

conda env list

conda create --name deepfaceid python=3.11

pip install -r requirements.txt

python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"

ipconfig

python run.py --debug --live --web-host=XXX


#### WSL:
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem \
  -out cert.pem \
  -days 365 \
  -subj "/CN=XXX" \
  -addext "subjectAltName=IP:XXX"
