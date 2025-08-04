# ASRSMA

## Installation
You can set up the environment using **Anaconda** 

```bash
git clone https://github.com/xxx/ASRSMA.git
cd ASRSMA
conda env create -f environment.yml
conda activate ASRSMA
```
<br>
## Dataset
The paths for the pre-trained data, fine-tuned data, and independent test set are `./ASRSMA/Dataset`.

<br>

## Running ASRSMA on your own dataset

### 1. Predict your own molecules with the trained model in batch mode:
#### Step 1: Preprocess the dataset using the following format:
```shell
example_data/
├── 1arj.PDB
│   ├── complex.pdb
│   ├── ligand.sdf
│   ├── pocket.txt
└── └── receptor.pdb
  
The ligand file can be in .sdf, .mol2, or .mol format. The pocket.txt contains the residues of binding pockets, each residue is named as follows:
```
A_7_U
A_64_G
A_132_C
...

#### Step 2: Run the following command to preprocess the dataset**
```bash
python preprocess.py --input_path [raw_path] --output_path [processed_path]
```

#### Step 3: Predict the dataset using the following command
```bash
python test.py --input_path [processed_path] --output_path [predict_path] 
```
Then you can find the prediction results in the `predict_path`.

#### Step 4: Predict the single dataset
```bash
python inference.py.py --input_path [data_path] --output_path [predict_path] 
```
<br>

## Reproducing paper numbers

Running the following command to reproduce the evaluation result on `Finetuning_data`:
```bash
python test.py --input_path dataset/Finetuning/test_data --output_path [test_path]
```
Running the following command to reproduce the evaluation result on `Independen_test`:
```bash
python test.py --input_path dataset/Independen_test --output_path [test_path]
```

## Training a model yourself
You can also retraining the model yourself by using the following command:
#### Step 1: Pre-train**
```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node pretrain.py --data_dir Dataset/Pertrain
```
#### Step 1: Finetuning**
```bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='0' torchrun --nproc_per_node train.py --data_dir Dataset/Finetuning
```

