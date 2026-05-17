# NLP Emoții + Sarcasm (XLM-R heads-only)

## Structură
- `src/` codul (data/model/train/infer)
- `data/` 

## Instalare
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# mac/linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 
Instalează PyTorch cu CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Dataset
Așteaptă un fișier GoEmotions în `data/`:
- TSV cu 3 coloane: `text<TAB>label_id<TAB>comment_id` **sau**
- CSV cu coloane `text,label_id,comment_id`.

Setează în `src/train.py` sau rulează cu parametrii din `Config`.

## Train
```bash
python -m src.train
```

## Infer
```bash
python -m src.infer
```

## Sarcasm
- `rule` (recomandat): sarcasm doar cu regulă (nu antrenează head-ul)
- `pseudo`: antrenează head-ul pe pseudo-label (demo)
- `off`: fără sarcasm
