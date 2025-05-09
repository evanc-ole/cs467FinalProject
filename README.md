# cs467FinalProject

## Text-Based Model Instructions

1. **Install requirements**  
   `pip install -r requirements.txt`

2. **Run baseline model**  
   `python3 take_first.py`

3. **Train final model**  
   `python3 main.py`

4. **Evaluate trained and saved model**  
   `python3 evaluate_model.py`

## Image-Based Model Instructions

In the Image-Based folder...

1. **Install requirements**  
   `pip install -r requirements.txt`

2. **Unpack images zip**  
   `Unpack archive.zip with the 500 HEIC images in the Image-Based folder, name folder "Archive"`

2. **Change images to jpg**  
   `python rename.py`

3. **Yolo Prep**  
   `python YoloPrep.py`

4. **Yolo Train**  
   `python YoloTrain.py`

5. **(optional) Yolo Inferences**  
   `python YoloInferences.py`

6. **Post Yolo Cropper**  
   `python PostYoloCropper.py`

7. **Digit Prediction**  
   `python DigitPred.py`