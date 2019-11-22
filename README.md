#  Lion
Text pair classification toolkit.

## Usage

 ### Preprocessing:
 1. Transform the dataset to the standard formation . We currently support snli, qnli and quoraqp. Please write your own transformation scripts for other dataset.  
 `python  lion/data/dataset_utils/quoraqp.py convert-dataset  --indir INDIR --outdir OUTDIR`
 
 2. Preprocess the dataset.  
 ` python lion/data/processor.py process-dataset --in_dir IN_DIR --out_dir OUT_DIR --splits ['train'|'dev'|'test'] 
 --tokenizer_name [spacy/bert/xlnet] --vocab_file FILE_PATH --max_length SEQUENCE_LENGTH`

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

Performance on the dev set
<table>
  <tr>
    <th width=25%, bgcolor=#999999 >Model</th> 
    <th width=25%, bgcolor=#999999>Quora QP</th>
    <th width="25%", bgcolor=#999999>SNLI</th>
    <th width="25%", bgcolor=#999999>QNLI</th>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> BiMPM </td>
    <td align="center", bgcolor=#eeeeee> 86.9(88.17) </td>
    <td align="center", bgcolor=#eeeeee> 86.0(86.9) </td>
    <td align="center", bgcolor=#eeeeee>  </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> Esim </td>
    <td align="center", bgcolor=#eeeeee> 88.4 </td>
    <td align="center", bgcolor=#eeeeee> 87.4 </td>
    <td align="center", bgcolor=#eeeeee>  </td>
  </tr>
<tr>
    <td align="center", bgcolor=#eeeeee> BERT </td>
    <td align="center", bgcolor=#eeeeee> 91.3</td>
    <td align="center", bgcolor=#eeeeee> 91.1 </td>
    <td align="center", bgcolor=#eeeeee> 91.7 </td>
  </tr>
  <tr>
    <td align="center", bgcolor=#eeeeee> XLNET </td>
    <td align="center", bgcolor=#eeeeee> 90.9   </td>
    <td align="center", bgcolor=#eeeeee> 91.6   </td>
    <td align="center", bgcolor=#eeeeee> 91.7  </td>
  </tr>
</table>

## License

[Apache-2.0](https://opensource.org/licenses/Apache-2.0)
