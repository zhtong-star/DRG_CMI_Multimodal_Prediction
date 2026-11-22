"""Offline integration check with synthetic CSVs and tiny randomly initialized encoders."""
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, LongformerConfig, LongformerModel, BertConfig, BertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from create_cohorts import main as prepare_cohort
from extract_structured import main as extract
from finalize_cohorts import finalize
from main_multimodal import main as run, load_fold
from data_utils_multimodal import MultimodalDrgDataset
from generate_report import make_case_report
from protocol import read_json, write_json

def build_fixture(root):
    mimic, rules = root/'mimic',root/'rules'
    mimic.mkdir(parents=True); rules.mkdir()
    admissions=[]; patients=[]; icu=[]; drg=[]; notes=[]; events=[]
    for subject in range(1,61):
        patients.append({'SUBJECT_ID':subject,'DOB':'2050-01-01 00:00:00'})
        for repeat in range(2 if subject % 10 == 0 else 1):
            hadm=1000+subject*10+repeat
            start=pd.Timestamp('2100-01-01')+pd.Timedelta(days=repeat*10)
            intime=start+pd.Timedelta(hours=2)
            admissions.append({'SUBJECT_ID':subject,'HADM_ID':hadm,'ADMITTIME':start,'DISCHTIME':start+pd.Timedelta(days=5)})
            icu.append({'SUBJECT_ID':subject,'HADM_ID':hadm,'ICUSTAY_ID':hadm+10000,'INTIME':intime,
                        'OUTTIME':intime+pd.Timedelta(hours=30 if subject%2 else 50),'LOS':2,
                        'FIRST_WARDID':1,'LAST_WARDID':1,'FIRST_CAREUNIT':'MICU','LAST_CAREUNIT':'MICU'})
            drg.append({'SUBJECT_ID':subject,'HADM_ID':hadm,'DRG_TYPE':'MS','DRG_CODE':(subject+repeat)%3+1,'DESCRIPTION':'test'})
            for hour,category,text in [(-1,'Nursing','patient [**Name**] has fever and cough early today'),
                                       (24.5,'Physician','later onset hypotension concern for infection today'),
                                       (2,'Discharge summary','must never enter early text')]:
                notes.append({'SUBJECT_ID':subject,'HADM_ID':hadm,'CHARTTIME':intime+pd.Timedelta(hours=hour),'CATEGORY':category,'TEXT':text})
            if subject != 60:  # Real exclusion path: both stays of this synthetic patient lack structured data.
                for hour in [.2,2.5,23.9,24.5,47.5,48.1]:
                    events.append({'SUBJECT_ID':subject,'HADM_ID':hadm,'ITEMID':220045,
                                   'CHARTTIME':intime+pd.Timedelta(hours=hour),'VALUENUM':70+hour+subject%3})
    for filename,rows in [('ADMISSIONS',admissions),('PATIENTS',patients),('ICUSTAYS',icu),('DRGCODES',drg),('NOTEEVENTS',notes),('CHARTEVENTS',events)]:
        pd.DataFrame(rows).to_csv(mimic/f'{filename}.csv',index=False)
    pd.DataFrame(columns=['SUBJECT_ID','HADM_ID','ITEMID','CHARTTIME','VALUENUM']).to_csv(mimic/'LABEVENTS.csv',index=False)
    pd.DataFrame({'DRG_CODE':[1,2,3], 'MS-DRG':[1,2,3],'MDC':[1,1,1],'WEIGHT':[1.,2.,3.],
                  'DESCRIPTION':['SEPSIS W/O CC/MCC','SEPSIS W CC','SEPSIS W MCC']}).to_csv(rules/'MSDRG_RULE13.csv',index=False)
    vocab={'<s>':0,'<pad>':1,'</s>':2,'<unk>':3}
    for word in 'patient has fever and cough early today later onset hypotension concern for infection'.split():
        if word not in vocab: vocab[word]=len(vocab)
    tokenizer=Tokenizer(WordLevel(vocab,unk_token='<unk>')); tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='<unk>',pad_token='<pad>',bos_token='<s>',eos_token='</s>')
    encoder=root/'tiny_longformer'; encoder.mkdir()
    fast.save_pretrained(encoder)
    LongformerModel(LongformerConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,
                    num_attention_heads=2,intermediate_size=32,max_position_embeddings=66,
                    attention_window=[8],pad_token_id=1,bos_token_id=0,eos_token_id=2)).save_pretrained(encoder)
    bert=root/'tiny_bert'; bert.mkdir()
    fast.save_pretrained(bert)
    BertModel(BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                        intermediate_size=32,max_position_embeddings=64,pad_token_id=1)).save_pretrained(bert)
    return mimic,rules,encoder,bert

def main():
    torch.set_num_threads(1)
    Path('.qa').mkdir(exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='smoke_', dir='.qa')).resolve()
    mimic,rules,encoder,bert=build_fixture(root)
    data=root/'data_n'; output=root/'results'
    prepare_cohort(['--mimic_dir',str(mimic),'--rule_dir',str(rules),'--data_dir',str(data)])
    extract(['--mimic_dir',str(mimic),'--data_dir',str(data),'--chunksize','31'])
    audit=finalize(data/'ms',rules/'MSDRG_RULE13.csv')
    assert audit['excluded_admissions']==2
    common=['--data_dir',str(data),'--rule_dir',str(rules),'--bert_model_name',str(encoder),
            '--output_dir',str(output),'--device','cpu','--epochs','1','--batch_size','16',
            '--max_seq_length','32','--struct_hidden_size','8','--struct_num_layers','1',
            '--cnn_filter_maps','8','--embedding_dim','8','--gradient_accumulation_steps','2','--eval_cc_mcc']
    run(common+['--model','LGGCMN','--logit_adjustment','0.7','--run_name','tiny_fivefold'])
    result=read_json(output/'tiny_fivefold'/'summary.json')
    assert result['status']=='complete'
    before=np.load(output/'tiny_fivefold'/'fold_0'/'eval_48h'/'logits.npz')['logits'].copy()
    run(['--eval_model',str(output/'tiny_fivefold'),'--device','cpu','--eval_cc_mcc'])
    after=np.load(output/'tiny_fivefold'/'fold_0'/'eval_48h'/'logits.npz')['logits']
    np.testing.assert_allclose(before,after,atol=1e-6)
    for model,extras in [('GRU',[]),('Longformer',[]),('Concat',[]),
                         ('CAML',['--allow_random_caml_embeddings','--word_min_freq','1'])]:
        run(common+['--model',model,'--fold','0','--run_name','tiny_'+model]+extras)
    run(common+['--model','LGGCMN','--logit_adjustment','0.7','--fold','0','--run_name','tiny_global','--text_pooling','global'])
    run(common+['--model','LGGCMN','--logit_adjustment','0.7','--fold','0','--run_name','tiny_concat_ablation','--fusion_method','concat'])
    run(common+['--model','LGGCMN','--logit_adjustment','0.7','--fold','0','--run_name','tiny_bert',
                '--text_backbone','bert','--bert_model_name',str(bert)])
    fold=output/'tiny_fivefold'/'fold_0'
    stay=pd.read_csv(fold/'test.csv').stay.iloc[0]
    report=make_case_report(fold,stay,48,'cpu')
    assert report['peak_interval_hours'][1]-report['peak_interval_hours'][0]==1
    assert report['peak_time_hours'] is not None
    assert 0 <= report['peak_time_hours'] < 48
    write_json(root/'case_report.json',report)
    write_json(root/'verification.json', {'synthetic':True,'fivefold_training':'passed','checkpoint_reload':'passed',
                                         'four_baselines':'passed','three_ablations':'passed','attribution':'passed',
                                         'excluded_incomplete_admissions':2})
    print('OFFLINE SYNTHETIC INTEGRATION PASSED; this is not reproduction of manuscript results.')

if __name__=='__main__':
    main()
