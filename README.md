# **CONORM**  
CONORM: Context-Aware Entity Normalization for Adverse Drug Event Detection

---

## **Developed with**

- **Operating System:** Ubuntu 22.04.3 LTS  
    - Kernel: Linux 4.18.0-477.27.1.el8_8.x86_64  
    - Architecture: x86_64  
- **Python:**  
    - 3.10.12  

---

## **Prerequisites**  

### **Install OpenJDK**  

Ensure `openjdk-8-jdk` is installed on your system, as it is required for BRATEval evaluation and the pipeline will not function without it.  

```bash  
# Update the Package Index.  
sudo apt update  

# Install OpenJDK 8.  
sudo apt install openjdk-8-jdk  

# Verify the Installation.  
java -version  
```  

### **Installation of Required Python Libraries**  

Install the necessary Python libraries using the following steps:  

```bash  
# Create the environment.  
conda create -n conorm python=3.10.12  

# Activate the environment.  
conda activate conorm

# Go to the project root folder.  
cd <path to the project root folder>  

# Install external libraries.  
pip install -r requirements.txt  

# Download punkt tokenizer  
python -m nltk.downloader punkt
```  

---

## **Structure**  

```bash  
├── config  
│   └── ner_training_config.json (configuration required for conorm-ner training)  
│   └── ner_inference_config.json (configuration required for conorm-ner inference)  
│   └── en_training_config.json (configuration required for conorm-en training)  
│   └── en_inference_config.json (configuration required for conorm-en inference)  
│  
├── data  
│   └── <lang> (folder for language-specific datasets)  
│       └── <dataset> (folder for dataset-specific files)  
│           └── train (standoff training data)  
│           └── val (standoff validation data)  
│           └── test (optional standoff testing data)  
│           └── infer (optional standoff inference files)  
│  
├── src (source code)  
│   ├── inference_helpers.py  
│   ├── model_helpers.py  
│   ├── conorm_en_inference_helpers.py  
│   ├── conorm_en_helpers.py  
│   └── standoff2bio.py  
│  
├── 0_preprocessing.py (preprocessing script for standoff data)  
├── 1_ner_train.py (training script for NER model)  
├── 2_ner_infer.py (inference script for trained NER model)
├── 3_en_train.py (training script for EN model)  
├── 4_en_infer.py (inference script for trained EN model)  
├── BRATEval-0.0.2-SNAPSHOT.jar (BRAT evaluation tool)  
├── compare_normalizations.py (NER+EN evaluation tool)  
└── requirements.txt (Python dependencies)  
```  

---

## **Typical Pipeline**  

### **Upload Data**  

The pipeline only accepts data in the [standoff data format](https://brat.nlplab.org/standoff.html). Ensure your data adheres to the following structure:  

```bash  
├── data  
│   └── <lang>  
│       └── <dataset>  
│           └── train (standoff training data)  
│           └── val (standoff validation data)  
│           └── test (optional: standoff testing data)  
```  

If the `test` folder is not provided, set `"test_standoff_path"` to `null` in `./config/ner_training_config.json`. A synthetic test folder will be created during preprocessing by copying validation files.

Some sample files have been added to this repository as input examples, located at `"./data/english/cadec"`.

---

### **NER Training Configuration**  

Edit the file `./config/ner_training_config.json` to configure training parameters

---

### **Step 0: Preprocessing**  

Run the preprocessing script to prepare the data:  

```bash  
python 0_preprocessing.py ./config/ner_training_config.json  
```  

---

### **Step 1: Model Training**  

Train the NER model using the training script:  

```bash  
python 1_ner_train.py ./config/ner_training_config.json  
```  

After training, the model and associated files (e.g., logs, evaluation metrics) will be saved in `./logs/<lang>_<dataset>/<model_name>_<date>_<time>`.  

---

### **Inference Configuration**  

Edit the file `./config/ner_inference_config.json` to configure inference parameters

---

### **Step 2: Inference**  

Run the inference script to apply the trained NER model:  

```bash  
python 2_ner_infer.py ./config/ner_inference_config.json  
```  

Predictions will be saved to the `output_path` specified in the inference configuration file.

### **EN Training Configuration**  

Edit the file `./config/en_training_config.json` to configure training parameters. Note that you must have access to the MedDRA data files (`llt.asc` and `pt.asc`) and provide their paths for EN model training/inference.

### **Step 1: Model Training**  

Train the EN model using the training script:  

```bash  
python 3_en_train.py ./config/en_training_config.json  
```  

After training, the model and associated files will be saved in `./logs/conormen/<lang>_<dataset>/<model_name>_<date>_<time>`.  

---

### **Inference Configuration**  

Edit the file `./config/en_inference_config.json` to configure inference parameters

---

### **Step 2: Inference**  

Run the inference script to apply the trained EN model:  

```bash  
python 4_en_infer.py ./config/en_inference_config.json  
```
