# ASL Detection Using CNN
## Project Description
Preprocess the ASL dataset with SAM2 in combination with MediaPipe and classify with customly trained CNN-RNN pipeline.

Final report in NeurIPS Paper format: ASL_Detection_Report.pdf
## ECE324 Project
-----
## Directory structure
NOTE: followed CCDS suggested file structure
* `README.md` - The top-level README for developers using this project.
  * data
    * external - Data from third party MS-ASL.
    * processed - The final, canonical data sets for modeling.
    * raw - The original, immutable data dump.
* notebooks - Jupyter notebooks.
  * demo - Dataset used for demo purpose.
  * `SAM2-initial-exploration.ipynb` - Initial SAM2 model exploraiton notebook.
  * `SAM2+MediaPipe-automasking.ipynb` - Final SAM2 + MediaPipe pipeline for automasking processing notebook.
* reports - Generated analysis as LaTeX.
  * figures - Generated graphics and figures to be used in reporting
* source_codes - Source code for use in this project.
    * `CNN.py`
    * `CNNRNN_v2.py`
    * processing 
      * `generate_batch.py`
      * `process_batch.py`
      * `split_batch.py`
      * `video2img.py`
    * `Image_Conversion.py`
    * `Image_to_tensor.py`
    * `visualize_model.py` - Code to create visualizations.
