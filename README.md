# CNN - Model Swapping con VGG16

## Descrizione
Progetto del modulo CNN del Master Epicode.

L'obiettivo era sostituire l'architettura MobileNetV2 con VGG16 per la classificazione di immagini utilizzando i pesi pre-addestrati di ImageNet.

## Modifiche effettuate
- Sostituzione di MobileNetV2 con VGG16
- Utilizzo del preprocess_input specifico di VGG16
- Cambio dell'immagine di input
- Visualizzazione delle Top-5 predizioni

## Tecnologie utilizzate
- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Requests

## Esecuzione

```bash
pip install -r requirements.txt
py main.py