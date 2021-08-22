# Machine learning on audio data

This project was carried out within the Python Programming course in Bioinformatics Institute.

The goal of the project was to implement different ML methods on audio data. This included **music genre classification**, **music segmentation** and **music clustering**.

## Description

The project was made through efforts of following contributors:

+ [Churkina Anna](https://github.com/AnyaChurkina) (music segmentation)
+ [Piankov Ivan](https://github.com/IvanPiankov) (music clustering)
+ [Litvinov Daniil](https://github.com/danon6868) (musical Tinder)
+ [Kirillova Ekaterina](https://github.com/MsKittin) (data preprocessing)
+ [Sidorin Anton](https://github.com/SidorinAnton) (music genre classification)
+ [Kruglikov Roman](https://github.com/krglkvrmn) (music genre classification)

Detailed description of each sub-project can be found in corresponding directory.

## Usage

### Download source code and install dependencies

```
git clone https://github.com/krglkvrmn/IB_Music_project
cd IB_Music_project
pip install -r requirements.txt
```

### Download additional data

```
# FMA dataset metadata
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -O data/fma_metadata.zip

# Small FMA dataset (8000 tracks, 8 genres, 30 seconds each)
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip -O data/fma_small.zip

cd data
unzip fma_small.zip
unzip fma_metadata.zip
```

### Run scripts and notebooks

```
jupyter notebook <notebook_name>
# or
jupyter lab
# or
python <script_name> [parameters_list]
```

## Dependencies

All library dependencies are listed in **requirements.txt** file. Scripts and notebooks were run on Python of at least 3.7 version. Stable work on earlier releases is not guaranteed. 
