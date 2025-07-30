# WE_KLT_Kontrolle
In dieser Arbeit wird ein Konzept zur automatisierten Belegungsüberprüfung von Kleinladungsträger (KLT) mittels Convolutional Neural Network (CNN)s vorgestellt. 
Ziel ist es, eine effiziente und zuverlässige Methode zu entwickeln, die es ermöglicht, falsch belegte KLTs frühzeitig zu identifizieren und somit die Pickprozesse vom Pick-and-Place-Roboter beim Versand zu verbessern.


ResNet18_train.py ist für modell training.

kamera_grab_with_barcode.py ist für bildersammlung.

final_script.py ist für echzeit Vohersage.


Python Umgebung erstellen:

Python >3.9

CMD-Befehle:

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

pip install --upgrade transformers

