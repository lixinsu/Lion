#  Lion
Text pair classification toolkit.

## Usage

 ### Preprocessing:
 1. Transform the dataset to the standard formation . We currently support snli and quoraqp. Please write your own transformation scripts for other dataset.  
 `python  lion/data/dataset_utils/quoraqp.py convert-dataset  --indir INDIR --outdir OUTDIR`
 
 2. Preprocess the dataset.  
 ` python lion/data/processor.py process-dataset --in-dir IN_DIR --out-dir OUT_DIR --tokenizer-name [spacy/bert] --vocab-file FILE_PATH`

 ### Training:  
1. Create a directory for saving model and put the config file in it . 
2. Edit the config file, modifying the train file and dev file path . 
3. Run **lion/training/trainer.py**   
For example:  
`python lion/training/trainer.py --train --output_dir experiments/QQP/esim/` . 

### Hyper-parameter searching
1. Create a directory for saving model and put the config file in it . 
2. Edit the config file, modifying the train file and dev file path . 
3. Edit the tuned_params.yaml 
For example:
```
hidden_size:
    - 100
    - 200
    - 300
dropout:
    - 0.1
    - 0.2
```
4. Run `python lion/training/search_parameter.py --parent_dir experiments/QQP/esim/hidden_dim/` 


 ### Evaluation:
 `python lion/training/trainer.py --evaluate --output_dir experiments/QQP/esim/ --dev_file your_dev_path`
 
 ### Testing:
`python lion/training/trainer.py --test --output_dir experiments/QQP/esim/ --test_file your_test_file`


## Models

<table>
  <tr>
    <th width=30%, bgcolor=#999999 >Model</th> 
    <th width=35%, bgcolor=#999999>Quora QP</th>
    <th width="35%", bgcolor=#999999>SNLI</th>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> BiMPM </td>
    <td align="center", bgcolor=#eeeeee> 86.9(88.17) </td>
    <td align="center", bgcolor=#eeeeee> 86.0(86.9) </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Esim </td>
    <td align="center", bgcolor=#eeeeee> 88.4 </td>
    <td align="center", bgcolor=#eeeeee> 87.4 </td>
  </tr>

</table>

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)
