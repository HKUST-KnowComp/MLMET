# MLMET
Ultra-Fine Entity Typing with Weak Supervision from a Masked Language Model

Requires transformers, inflect.

1. Set DATA_DIR in config.py to your data directory

2. Put English Gigaword data to DATA_DIR/gigaword_eng_5
https://catalog.ldc.upenn.edu/LDC2011T07
   
3. Put **full** Ultra-fine Entity Typing data to DATA_DIR/ultrafine/uf_data
https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html

4. Create the following directories:
DATA_DIR/ultrafine/output/models, DATA_DIR/ultrafine/log, DATA_DIR/ultrafine/bert_labels

5. Run 
   ```python prep.py``` 
   to generate pronoun mentions.
   
6. Run
```python trainweak.py 0```
   to pretrain a model with existing weak samples.
   
7. Run
```python trainbertuf.py 0```
   to finetune the model trained in 6.
   
8. Run
```python genlabels.py```
   to generate labels with BERT MLM. This can take a long time. You may want to adjust the code to do it in parallel. 
   
   Pre-generated labels for non-pronoun mention examples:
https://drive.google.com/file/d/1FeaPyIovdkkumVZteeNWyWYErCd6kbB1/view?usp=sharing
   
   Pre-generated labels for pronoun mention examples are in data/gigaword5_pronoun_s005_ama_ms_10types.zip . Before using it, run ```python verifypronoun.py``` to verify if your pronoun mentions match these labels.
   If ```python verifypronoun.py``` prints ```Verification FAILED!```, try running ```python genpronfixed.py``` to generate pronoun mention examples.

9. Run
```python trainweak.py 1```
   to pretrain a model with newly generated weak labels.
   
10. Run 
    ```python trainbertuf.py 1```
    to fine-tune the model in 9.

11. Run
```python trainufst.py```
    to do self-training.
    
