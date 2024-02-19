# Proembryo detection
---

# Modelo para predicción

Graphic User Interface (GUI) supported by `Tkinter` running a simple python program for micrographs /cell image vision detection, classification and annotation of cell types (Proembryos derived from Somatic Embryogenesis)

- **Input** : 
 1) Folder path with subfolders up to 3 levels (Filenames without hyphens "-")
 2) Model for proembryo detection for your species in `/models`
 3) File path directory for output

*EXAMPLE of contents in input folder*
 ```
 .
└── Filepath_example (<-- input for the program)/
    ├── experiment 1 (e.g. control) - 2 levels/
    │   ├── rep1/
    │   │   └── picture_1.png/tif/
    │   │       ├── picture_2.png/tif
    │   │       └── picture_3.png/tif
    │   ├── rep2/
    │   │   ├── picture_1.png/tif
    │   │   ├── picture_2.png/tif
    │   │   └── picture_3.png/tif
    │   └── rep3/
    │       ├── picture_1.png/tif
    │       ├── picture_2.png/tif
    │       └── picture_3.png/tif
    └── experiment 2 - 3 levels/
        ├── Concentration 1/
        │   ├── rep1/
        │   │   └── picture_1.png/tif/
        │   │       ├── picture_2.png/tif
        │   │       └── picture_3.png/tif
        │   ├── rep2/
        │   │   ├── picture_1.png/tif
        │   │   ├── picture_2.png/tif
        │   │   └── picture_3.png/tif
        │   └── rep3/
        │       ├── picture_1.png/tif
        │       ├── picture_2.png/tif
        │       └── picture_3.png/tif
        └── Concentration 2/
            ├── rep1/
            │   └── picture_1.png/tif/
            │       ├── picture_2.png/tif
            │       └── picture_3.png/tif
            ├── rep2/
            │   ├── picture_1.png/tif
            │   ├── picture_2.png/tif
            │   └── picture_3.png/tif
            └── rep3/
                ├── picture_1.png/tif
                ├── picture_2.png/tif
                └── picture_3.png/tif
 ```

 - **Output** : 
 1) Barplots of proembryo percentage and cell prediction type mean frequencies, with printed std
 2) Frequencies of cell classes per picture / experiment etc
 3) Folders with the structure of Folder input with input pictures, annotated 

Example Folder Tree structure:


# How to run

Download and install [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/) for windows/linux etc

Once installed add `miniconda` path were the program was installed to PATH variable, open CMD / Terminal in windows and type:

```
conda install --file requirements.txt

```
If Linux is used, open terminal and type

```
python3 `path_to\CONTEO_Microspora_Proembrión`

```
If Windows is used, you can create a batch file that will execute the program without upon opening the windows CMD terminal, with these contents.(save as .bat) - Functions if installed for a single user

```
@echo on
call "C:\Users\{your_username}\miniconda3\Scripts\activate.bat"
"C:\Users\{your_username}\miniconda3\python.exe" "path_to\CONTEO_Microspora_Proembrion.py"

```



| `File/Folder` | `Description`  |
|---|---|
| `models`  | Folder with Pickle models to load in the program for prediction of proembryos |
| `CONTEO_Microspora_Proembrion.py`  | python script with functionality to train your own unsupervised model using a file path with pictures/ predict using model example .pkl file in `/models` | 

