# First we install all the prerequisites to be able to download from Zenodo and Google Drive
python3 -m venv env
source env/bin/activate
pip install zenodo_get gdown
git lfs install


### Download the datasets
### AnimalSpeak
echo "Downloading AnimalSpeak"
git clone https://huggingface.co/datasets/davidrrobinson/AnimalSpeak
echo "AnimalSpeak downloaded"

#### AudioCaps
echo "Downloading AudioCaps"
git clone https://huggingface.co/datasets/jp1924/AudioCaps
echo "AudioCaps downloaded"

### AudioCaption
echo "Downloading AudioCaption"
mkdir AudioCaption && cd AudioCaption && zenodo_get 4671263 && cd ..
echo "AudioCaption downloaded"

### AudioDiffCaps
echo "Downloading AudioDiffCaps"
git clone https://github.com/nttcslab/audio-diff-caps
# TODO: Perform instructions in README.
echo "AudioDiffCaps downloaded"

### AudioGrounding
echo "Downloading AudioGrounding"
mkdir AudioGrounding && cd AudioGrounding && zenodo_get 7269161 && cd ..
echo "AudioGrounding downloaded"

### AudioSet
echo "Downloading AudioSet"
git clone https://huggingface.co/datasets/agkphysics/AudioSet
echo "AudioSet downloaded"

### CAPTDURE
echo "Downloading CAPTDURE"
mkdir CAPTDURE && cd CAPTDURE && zenodo_get 7965763 && cd ..
echo "CAPTDURE downloaded"

### Clotho
echo "Downloading Clotho"
mkdir Clotho && cd Clotho && zenodo_get 3490684 && cd ..
echo "Clotho downloaded"

### ClothoDetail
echo "Downloading ClothoDetail"
mkdir ClothoDetail && cd ClothoDetail && wget https://huggingface.co/datasets/magicr/BuboGPT/raw/main/Clotho-detail-annotation.json && cd ..
echo "ClothoDetail downloaded"

### ClothoAQA
echo "Downloading ClothoAQA"
mkdir ClothoAQA && cd ClothoAQA && zenodo_get 6473207 && cd ..
echo "ClothoAQA downloaded"

### DAQA
echo "Downloading DAQA"
git clone https://github.com/facebookresearch/daqa
### TODO: Perform instructions in README.
echo "DAQA downloaded"

### FAVDBench 
#### apply for dataset at https://github.com/OpenNLPLab/FAVDBench

### FSD50k
echo "Downloading FSD50k"
mkdir FSD50k && cd FSD50k && zenodo_get 4060432 && cd ..
echo "FSD50k downloaded"

### MACS
echo "Downloading MACS"
mkdir MACS && cd MACS && zenodo_get 5114771 && cd ..
echo "MACS downloaded"

### mClothoAQA
echo "Downloading mClothoAQA"
git clone https://github.com/swarupbehera/mAQA/
### TODO: Perform instructions in README.
echo "mClothoAQA downloaded"

### MULTIS
echo "Downloading MULTIS"
mkdir MULTIS && cd MULTIS && gdown https://drive.google.com/uc?id=1C7k8flfITJ1GxMwFSvEmBFGyevDZl1ke
echo "MULTIS downloaded"

### SoundDescs
echo "Downloading SoundDescs"
git clone https://github.com/akoepke/audio-retrieval-benchmark
# TODO: Perform instructions in README.
echo "SoundDescs downloaded"

### SoundingEarth
echo "Downloading SoundingEarth"
mkdir SoundingEarth && cd SoundingEarth && zenodo_get 5600379 && cd ..
echo "SoundingEarth downloaded"

### VGGSound
echo "Downloading VGGSound"
git clone https://huggingface.co/datasets/Loie/VGGSound
echo "VGGSound downloaded"

### WavText5k
echo "Downloading WavText5k"
git clone https://github.com/microsoft/WavText5K
### TODO: Perform instructions in README.
echo "WavText5k downloaded"