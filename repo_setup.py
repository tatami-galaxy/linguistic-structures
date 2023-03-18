import os
import gdown

os.mkdir('data')
os.mkdir('data/raw')
os.mkdir('data/processed')
os.mkdir('data/interim')
os.mkdir('data/processed/UD')
os.mkdir('data/processed/wikiann')

#os.mkdir('models')
#os.mkdir('models/probes')
#os.mkdir('models/finetuned')

url = "https://drive.google.com/drive/folders/18R3U5J6XvZusJoqbg8cIX2j2Dtgz6Lqm?usp=sharing"
gdown.download_folder(url, quiet=True, use_cookies=False)

