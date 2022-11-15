
# AEG

Code and dataset for our EMNLP 2022 paper: 

AEG: Argumentative Essay Generation via A Dual-Decoder Model with Content Planning

# Prerequisites

```
python==3.7.11
datasets==2.5.2  
nltk==3.6.5  
numpy==1.21.6  
torch==1.9.0+cu111  
transformers==4.14.1  
```

# Usage

```
python get_tfidf_keywords.py
python preprocess.py
bash run.sh
```

See the results in `./outputs`.  
For the pre-training results, please first train the model with the [CNN-DailyMail](https://huggingface.co/datasets/ccdv/cnn_dailymail) dataset with the same procedure.
