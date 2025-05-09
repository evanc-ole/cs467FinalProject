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

2. **Download Images**  
   `Visit drive link with USC email (https://drive.google.com/drive/folders/1mQlhYis6gyfyMfApv4oL-BfXQ5aNuFkJ?usp=sharing) and download Archive.zip`

3. **Unpack images zip**  
   `Unpack archive.zip with the 500 HEIC images in the Image-Based folder, name folder "Archive"`

4. **Change images to jpg**  
   `python rename.py`

5. **Yolo Prep**  
   `python YoloPrep.py`

6. **Yolo Train**  
   `python YoloTrain.py`

7. **(optional) Yolo Inferences**  
   `python YoloInferences.py`

8. **Post Yolo Cropper**  
   `python PostYoloCropper.py`

9. **Digit Prediction**  
   `python DigitPred.py`
