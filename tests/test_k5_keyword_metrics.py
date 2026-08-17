"""Focused K5B.3 frozen-evidence and metrics tests."""
from __future__ import annotations
import copy
from scripts import k5_keyword_metrics as metrics

def _payload():
    papers=[]
    for i in range(10):
        papers.append({'paper_code':f'P{i+1:02d}','review_stage':'pilot' if i<2 else 'headline',
          'concepts':['one','two','three',None,None],'reviewer_notes':None,'excluded_from_headline_metrics':i<2})
    candidates=[]
    for i in range(92):
        candidates.append({'candidate_id':f'C{i:03d}','paper_code':f'P{(i%10)+1:02d}','review_stage':'pilot' if i%10<2 else 'headline',
          'decision':'accept','rejection_reason':None,'matched_concept_ids':'C1','confidence':'high','reviewer_notes':None})
    data={'schema_version':metrics.FROZEN_SCHEMA,'status':'frozen_complete_annotation','frozen_at':'x','reviewer_type':metrics.REVIEWER_TYPE,
      'bindings':{},'pilot_codes':['P01','P02'],'headline_codes':[f'P{i:02d}' for i in range(3,11)],'papers':papers,'candidates':candidates}
    data['frozen_annotation_sha256']=metrics.payload_hash(data);return data

def test_frozen_annotation_self_validates_and_excludes_pilots():
    assert metrics.validate_frozen_payload(_payload()) == []

def test_frozen_annotation_tampering_fails():
    data=_payload();data['candidates'][0]['decision']='reject'
    assert any('self-hash' in item for item in metrics.validate_frozen_payload(data))

def test_uncertain_is_not_in_resolved_precision_and_coverage_is_unique():
    rows=[
      {'paper_code':'P03','decision':'accept','rejection_reason':None,'matched_concept_ids':'C1, C2'},
      {'paper_code':'P03','decision':'accept','rejection_reason':None,'matched_concept_ids':'C2'},
      {'paper_code':'P03','decision':'reject','rejection_reason':'fragment','matched_concept_ids':None},
      {'paper_code':'P03','decision':'uncertain','rejection_reason':None,'matched_concept_ids':None},
    ]
    stats=metrics.method_stats(rows,['P03'],{'P03':['a','b','c',None,None]})
    assert stats['acceptance_rate']==0.5
    assert stats['resolved_precision']==round(2/3,6)
    assert stats['uncertain_rate']==0.25
    assert stats['macro_average_concept_coverage']==round(2/3,6)

def test_many_to_many_judgment_can_be_counted_for_each_method():
    judgment={'paper_code':'P03','decision':'accept','rejection_reason':None,'matched_concept_ids':'C1'}
    reference=metrics.method_stats([copy.deepcopy(judgment)],['P03'],{'P03':['a','b','c',None,None]})
    production=metrics.method_stats([copy.deepcopy(judgment)],['P03'],{'P03':['a','b','c',None,None]})
    assert reference['accept']==production['accept']==1
