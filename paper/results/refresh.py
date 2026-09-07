"""Refresh the three empirical tables and seven figure assets, preserving layout."""
from pathlib import Path
import hashlib,json,re,shutil
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PAPER=REPO/'paper'
REPORT=REPO/'results/revision_v20/_publication'
ALGORITHMS=['dqn','ddpg','td3','sac']
inputs={};outputs={}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 inputs[str(p.relative_to(REPO))]=digest(p)
 return pd.read_csv(p)
def fmt(x,n=2):return f'{0.0 if abs(x)<.5*10**(-n) else x:.{n}f}'
def replace_rows(text,label,start,rows):
 position=text.index('\\label{'+label+'}')
 begin=text.rfind('\\begin{table}',0,position)
 end=text.index('\\end{table}',position)
 block=text[begin:end]
 a=block.index(start);b=block.index('\\bottomrule',a)
 return text[:begin]+block[:a]+'\n'.join(rows)+'\n'+block[b:]+text[end:]

main=(PAPER/'main.tex').read_text()
s=read(HERE/'shortfall_summary.csv').set_index(['market','policy'])
rows=[]
for i,algo in enumerate([*ALGORITHMS,'as','twap']):
 if i==4:rows.append(r'\midrule')
 vals=[]
 for market in ('Synthetic','Historical'):
  r=s.loc[market,algo]
  vals += ['$'+fmt(r.IS)+'$', '$'+fmt(r.IS_lambda)+r'\;['+fmt(r.ci_low)+','+fmt(r.ci_high)+']$']
 rows.append(algo.upper()+' & '+' & '.join(vals)+r' \\')
main=replace_rows(main,'tab:economic_performance','DQN &',rows)

f=read(HERE/'forecast_summary.csv').set_index(['market','phase'])
rows=[]
for i,market in enumerate(['Synthetic','CAT','GOOGL','JPM','MSFT','PG']):
 if i:rows.append(r'\addlinespace[4pt]')
 for phase in ('clob','auction'):
  r=f.loc[market,phase]
  row=[market if phase=='clob' else '', 'CLOB' if phase=='clob' else 'Auction',
       fmt(r.h_mae,4),fmt(r.mid_mae,4),f'{r.mae_gain:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]',
       fmt(r.h_rmse,4),fmt(r.mid_rmse,4)]
  rows.append(' & '.join(row)+r' \\')
main=replace_rows(main,'tab:forecast_accuracy','Synthetic & CLOB',rows)

# Preserve the existing contrast order and use the validated report's intervals.
t=read(REPORT/'tables/treatments.csv')
audit=read(REPORT/'audit/treatments_by_seed.csv')
labels=dict(audit[['contrast_key','contrast']].drop_duplicates().values)
blocks=[('auction_credit',r'Dense versus sparse credit\\E-F versus E-S'),
        ('h_feature',r'H access\\E-F versus E-0'),
        ('h_anchor',r'Indicative versus frozen-mid anchor\\J-I versus J-F'),
        ('combined_preferences',r'Preferences, frozen-mid anchor\\J-F versus E-F'),
        ('auction_access',r'Retrained auction access\\E-0 versus No-A'),
        ('cashflow',r'Complete scheme versus raw cash flow\\J-I versus Cash-I'),
        ('economic_dense_h',r'Preferences, indicative anchor\\J-I versus E-I')]
rows=[]
for i,(key,label) in enumerate(blocks):
 if i:rows.append(r'\midrule' if i==5 else r'\addlinespace[4pt]')
 if i<5:g=t[t.contrast==labels[key]].set_index('algorithm');field='mean'
 else:g=read(REPO/f'results/revision_v20_{key}/_comparison/comparison.csv').set_index('algo');field='difference_bps'
 assert len(g)==4
 rows.append(r'\shortstack[l]{'+label+'}')
 for j,algo in enumerate(ALGORITHMS):
  r=g.loc[algo]
  rows.append(r'& \shortstack{$'+fmt(r[field])+r'$\\\scriptsize $['+fmt(r.ci_low)+','+fmt(r.ci_high)+']$}'+(r' \\' if j==3 else ''))
main=replace_rows(main,'tab:treatments',r'\shortstack[l]{Dense versus sparse credit',rows)
(PAPER/'main.tex').write_text(main)

figures={name:REPORT/f'figures/{name}.pdf' for name in (
 'economic_performance','auction_mechanism','treatments','credit_assignment','learning')}
figures.update(cashflow_comparison=REPO/'results/revision_v20_cashflow/_comparison/comparison.pdf',
               preferences_comparison=REPO/'results/revision_v20_economic_dense_h/_comparison/comparison.pdf')
for name,source in figures.items():
 assert source.read_bytes().startswith(b'%PDF-'),source
 destination=PAPER/f'figures/{name}.pdf.txt'
 inputs[str(source.relative_to(REPO))]=digest(source)
 shutil.copyfile(source,destination)
 assert digest(destination)==digest(source)
 outputs[str(destination.relative_to(REPO))]=dict(source=str(source.relative_to(REPO)),sha256=digest(destination))
(HERE/'refresh_manifest.json').write_text(json.dumps(dict(campaign='revision_v20',inputs=inputs,figure_outputs=outputs,
 tables=['tab:economic_performance','tab:forecast_accuracy','tab:treatments'],
 note='Numbers use revised runs only. The PDF .txt suffix is retained for compatibility with the supplied LaTeX source.'),indent=2)+'\n')
print('Refreshed three tables and seven figures from v20 results.')
