# ASL Detection Using CNN
## Project Description
ECE324 Project
-----
## Directory structure
NOTE: followed CCDS suggested file structure
├── README.md               <- The top-level README for developers using this project.
├── data
│   ├── external            <- Data from third party MS-ASL.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── notebooks               <- Jupyter notebooks.
|   ├── demo                <- Dataset used for demo purpose.
|   ├── SAM2-initial        <- Initial SAM2 model exploraiton notebook.
|   └── SAM2+MediaPipe      <- Final SAM2 + MediaPipe masking processing notebook.
│
├── reports                 <- Generated analysis as LaTeX.
│   └── figures             <- Generated graphics and figures to be used in reporting
|
└── source_codes            <- Source code for use in this project.
    |
    ├── CNN.py
    │
    ├── CNNRNN_v2.py
    │
    ├── processing 
    │   ├── generate_batch.py 
    │   ├── process_batch.py      
    │   ├── split_batch.py      
    │   └── video2img.py
    |
    ├── Image_Conversion.py
    │
    ├── Image_to_tensor.py
    |
    └── visualize_model.py  <- Code to create visualizations.
