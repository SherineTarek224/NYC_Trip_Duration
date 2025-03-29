# NYC Trip_Duration 
  ## project_Overview:
   - A Machine Learning model to predict total   trip duration of taxi  in New York city  using kaggle dataset .The model use this dataset which contains different features  and information
  
  ## Dependencies:
- **Install Miniconda/Anaconda**  
   - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight) or [Anaconda](https://www.anaconda.com/download) (full bundle).

- **Clone this repository**  
   ```bash
   git clone https://github.com/SherineTarek224/NYC_Trip_Duration.git
   cd NYC_Trip_Duration
- **Create Conda Environment**
    ```bash
    conda env create -f environment.yml
- **Activate the environment**
    ```bash
    conda activate Trip_duration_predict
- **Verify installation**
     ```bash
     conda list
## **Repo Structure**
   - ### **NYC_Trip_Duration**/

    ├── 📄README.md
    ├── 📄environment.yml
    ├── 📓(EDA).ipynb
    ├── 📜prepare.py
    ├── 📜preprocessing.py
    ├── 📜model.py
    ├── 📄sample_data_model.pkl
    ├── 📄model.pkl
    ├── 📂Dataset
       ├── 📂 Sample_Data
          ├── train_sample.csv
          ├── val_sample.csv
          ├── test_sample.csv
       ├── 📂 Data
          ├──train.zip
          ├── val.zip

- README.md:Contains information about this project and important details
- environment.yml:contain configuration of conda environment and library names and its version 
- EDA:Data analysis using jupyter notebook to explore features of dataset
- prepare.py:python file to prepare my data before preprocessing 
- preprocessing.py:python script to preprocess data choose different preprocessor and check outliers.Create preprocessing pipeline
- model.py:training the model (main)
- sample_data_model.pkl:Save model which trained and validation on sample dataset in pkl format
- model.pkl:Save model which trained on the whole dataset in pkl format 
- Dataset/sample_data:Dictionary which contain sample data and whole data
- - train_sample.csv:csv file contain training data 
- - val_sample.csv:csv file contain validation data 
- - test_sample.csv:csv file contain test data 
- Dataset/Data:
- - train.zip:contain the whole training dataset in zip format 
- - val.zip:contain the whole validation dataset in zip format

# Notes:
## Data Pipeline:
 - Splitting Data into numerical and categorical Features
 - Log Target Feature 
 - One hot encoding for categorical Features
 - Remove outliers using Z_Score and clipping method
 - Standard scaling for numerical features 
 - Polynomial Features (degree=6)
## Results:
### using Sample_Data

- Training RMSE polynomial_regression of degree 6 = 0.4787394095242249
- Training R2 polynomial_regression of degree 6 = 0.651067770855001
- Test RMSE polynomial_regression of degree 6 = 0.42408821507202693
- Test R2 polynomial_regression of degree 6 = 0.6986569865892237
### Using The whole Dataset
- Training RMSE polynomial_regression of degree 6 = 0.4535068868988568
- Training R2 polynomial_regression of degree 6 = 0.6750473324898938
- Test RMSE polynomial_regression of degree 6 = 0.4532444274284575
- Test R2 polynomial_regression of degree 6 = 0.6770645639917381

   

    
     
      

  
  
 