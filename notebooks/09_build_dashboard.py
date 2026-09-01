# Payments Fraud & Risk Decisioning Engine - Gagandeep Kapoor (2026)
"""Build 5-tab HTML dashboard from JSON outputs."""
import json, os
ROOT = "W:/My Documents/Shortcuts & Files/Fintech Fraud/github"
files = {'data_quality':'01_data_quality.json','eda':'02_eda.json',
         'sql':'04_sql_analysis.json','rules':'05_rule_engine.json',
         'ml':'06_ml_results.json','strategy':'07_threshold_strategy.json',
         'explainability':'08_explainability_monitoring.json'}
D = {}
for k,fn in files.items():
    with open(f"{ROOT}/outputs/{fn}") as f: D[k]=json.load(f)
dj = json.dumps(D, default=str)

CSS = """
:root{--bg:#f8fafc;--card:#ffffff;--bdr:#e2e8f0;--t1:#1e293b;--t2:#475569;--t3:#64748b;
--blue:#2563eb;--cyan:#0891b2;--grn:#059669;--red:#dc2626;--amb:#d97706;--pur:#7c3aed}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--t1);line-height:1.5}
.hdr{background:linear-gradient(135deg,#1e3a5f,#2563eb);border-bottom:1px solid var(--bdr);padding:20px 32px}
.hdr h1{font-size:1.5rem;font-weight:700;color:#fff}.hdr p{color:rgba(255,255,255,0.8);font-size:.85rem;margin-top:4px}
.hdr .author{color:rgba(255,255,255,0.7);font-size:.75rem;margin-top:2px}
.tabs{display:flex;background:var(--card);border-bottom:1px solid var(--bdr);padding:0 24px;overflow-x:auto}
.tab{padding:12px 20px;cursor:pointer;color:var(--t3);font-size:.82rem;font-weight:500;border-bottom:2px solid transparent;white-space:nowrap}
.tab:hover{color:var(--t1)}.tab.active{color:var(--blue);border-bottom-color:var(--blue)}
.tc{display:none;padding:24px}.tc.active{display:block}
.g{display:grid;gap:16px}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:1fr 1fr 1fr}.g4{grid-template-columns:1fr 1fr 1fr 1fr}
.c{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.ct{font-size:.75rem;color:var(--t3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
.kpi{text-align:center;padding:20px 16px}.kpi .v{font-size:1.8rem;font-weight:700;color:var(--t1);line-height:1.2}
.kpi .l{font-size:.75rem;color:var(--t3);margin-top:4px}.kpi .d{font-size:.8rem;margin-top:4px}
.cw{position:relative;width:100%}.cw canvas{width:100%!important}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--bdr)}
th{color:var(--t3);font-weight:600;text-transform:uppercase;font-size:.7rem;letter-spacing:.05em}td{color:var(--t2)}
tr:hover td{background:rgba(37,99,235,0.04)}
.sg{margin:12px 0}.sg label{display:block;font-size:.8rem;color:var(--t2);margin-bottom:4px}
.sg input[type=range]{width:100%;accent-color:var(--blue)}.sv{font-size:.85rem;color:var(--blue);font-weight:600}
.ftr{text-align:center;padding:20px;color:var(--t3);font-size:.75rem;border-top:1px solid var(--bdr)}
@media(max-width:768px){.g2,.g3,.g4{grid-template-columns:1fr}}
"""

BODY = """
<div class="hdr"><h1>Payments Fraud &amp; Risk Decisioning Engine</h1>
<p>Real-time fraud detection: 500K synthetic transactions | 3 ML models | Cost-sensitive 3-way decisioning</p>
<div class="author">Built by Gagandeep Kapoor</div></div>
<div class="tabs">
<div class="tab active" onclick="sw(0)">Overview</div>
<div class="tab" onclick="sw(1)">Fraud Patterns</div>
<div class="tab" onclick="sw(2)">Model Performance</div>
<div class="tab" onclick="sw(3)">Decision Strategy</div>
<div class="tab" onclick="sw(4)">Threshold Simulator</div>
</div>
<div class="tc active" id="t0">
 <div class="g g4" id="k0"></div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Daily Fraud Rate (%)</div><div class="cw"><canvas id="c01"></canvas></div></div>
  <div class="c"><div class="ct">Monthly Transaction Volume</div><div class="cw"><canvas id="c02"></canvas></div></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Fraud Rate by Category</div><div class="cw"><canvas id="c03"></canvas></div></div>
  <div class="c"><div class="ct">Fraud Rate by Payment Method</div><div class="cw"><canvas id="c04"></canvas></div></div>
 </div>
 <div class="c" style="margin-top:16px"><div class="ct">Data Quality Summary</div><div id="dq"></div></div>
</div>
<div class="tc" id="t1">
 <div class="g g2">
  <div class="c"><div class="ct">Hourly Fraud Rate</div><div class="cw"><canvas id="c11"></canvas></div></div>
  <div class="c"><div class="ct">Day of Week Fraud Rate</div><div class="cw"><canvas id="c12"></canvas></div></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Amount Distribution: Fraud vs Legitimate</div><div class="cw"><canvas id="c13"></canvas></div></div>
  <div class="c"><div class="ct">Fraud Rate by Amount Bucket</div><div class="cw"><canvas id="c14"></canvas></div></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Fraud by Country</div><div class="cw"><canvas id="c15"></canvas></div></div>
  <div class="c"><div class="ct">Fraud vs Legit Amount by Category</div><div class="cw"><canvas id="c16"></canvas></div></div>
 </div>
 <div class="c" style="margin-top:16px"><div class="ct">Top 20 Fraud Customers</div><table id="tf"></table></div>
</div>
<div class="tc" id="t2">
 <div class="g g3" id="k2"></div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">ROC Curves</div><div class="cw"><canvas id="c21"></canvas></div></div>
  <div class="c"><div class="ct">Precision-Recall Curves</div><div class="cw"><canvas id="c22"></canvas></div></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Feature Importance (Top 15)</div><div class="cw"><canvas id="c23"></canvas></div></div>
  <div class="c"><div class="ct">Score Distribution</div><div class="cw"><canvas id="c24"></canvas></div></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Rule Engine Performance</div><table id="rt"></table></div>
  <div class="c"><div class="ct">Model Comparison (Test Set)</div><table id="mt"></table></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">False Positive Profile</div><div id="fpp"></div></div>
  <div class="c"><div class="ct">False Negative Profile</div><div id="fnp"></div></div>
 </div>
</div>
<div class="tc" id="t3">
 <div class="g g4" id="k3"></div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Strategy Cost Comparison</div><div class="cw"><canvas id="c31"></canvas></div></div>
  <div class="c"><div class="ct">Fraud Prevention vs Customer Friction</div><div class="cw"><canvas id="c32"></canvas></div></div>
 </div>
 <div class="c" style="margin-top:16px"><div class="ct">Strategy Detail</div><table id="st"></table></div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Cost Breakdown by Strategy</div><div class="cw"><canvas id="c33"></canvas></div></div>
  <div class="c"><div class="ct">Monthly Model Drift</div><div class="cw"><canvas id="c34"></canvas></div></div>
 </div>
</div>
<div class="tc" id="t4">
 <div class="c">
  <div style="font-size:.95rem;font-weight:600;color:#fff;margin-bottom:12px">Interactive Threshold Simulator</div>
  <p style="color:var(--t2);font-size:.85rem;margin-bottom:16px">Adjust review and decline thresholds to explore the trade-off between fraud prevention, customer friction, and total cost.</p>
  <div class="g g2">
   <div>
    <div class="sg"><label>Review Threshold: <span class="sv" id="vr">0.10</span></label><input type="range" id="sr" min="0.01" max="0.90" step="0.01" value="0.10" oninput="usim()"></div>
    <div class="sg"><label>Decline Threshold: <span class="sv" id="vd">0.50</span></label><input type="range" id="sd" min="0.02" max="0.95" step="0.01" value="0.50" oninput="usim()"></div>
   </div>
   <div class="g g2" id="sk" style="gap:8px"></div>
  </div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Threshold vs Total Cost</div><div class="cw"><canvas id="c41"></canvas></div></div>
  <div class="c"><div class="ct">Threshold vs Precision / Recall</div><div class="cw"><canvas id="c42"></canvas></div></div>
 </div>
 <div class="g g2" style="margin-top:16px">
  <div class="c"><div class="ct">Decision Split</div><div class="cw"><canvas id="c43"></canvas></div></div>
  <div class="c"><div class="ct">Fraud Capture vs FPR</div><div class="cw"><canvas id="c44"></canvas></div></div>
 </div>
</div>
<div class="ftr">Payments Fraud &amp; Risk Decisioning Engine &copy; 2026 Gagandeep Kapoor | Warwick MEng Mechanical Engineering<br>Synthetic dataset for portfolio demonstration. Full pipeline: github.com/GagandeepSK/fintech-payments-risk-engine</div>
"""

# Write JS to a separate temp file then read it
JS = r"""
function sw(i){document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',j===i));
document.querySelectorAll('.tc').forEach((t,j)=>t.classList.toggle('active',j===i));}
const fmt=n=>typeof n==='number'?n.toLocaleString():n;
const gbp=n=>'\u00A3'+Math.round(n).toLocaleString();
function mkKPI(id,items){document.getElementById(id).innerHTML=items.map(k=>`<div class="c kpi"><div class="v" style="color:${k.c||'#fff'}">${k.v}</div><div class="l">${k.l}</div>${k.d?'<div class="d" style="color:var(--t2)">'+k.d+'</div>':''}</div>`).join('');}
const co={responsive:true,plugins:{legend:{labels:{color:'#475569',font:{size:11}}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}},y:{ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}}}};
function mc(c,type,labels,datasets,opts){return new Chart(c,{type,data:{labels,datasets},options:{...JSON.parse(JSON.stringify(co)),...(opts||{})}});}

// TAB 0
const ov=D.sql.overall_summary[0];
mkKPI('k0',[{v:fmt(ov.total_txns),l:'Total Transactions',c:'#3b82f6'},{v:fmt(ov.fraud_txns),l:'Fraud Transactions',c:'#ef4444'},{v:ov.fraud_rate_pct+'%',l:'Fraud Rate',c:'#f59e0b'},{v:gbp(ov.fraud_value),l:'Fraud Value',c:'#ef4444'}]);

mc('c01','line',D.eda.daily_fraud.dates.filter((_,i)=>i%3===0),[{label:'Fraud Rate %',data:D.eda.daily_fraud.rate.filter((_,i)=>i%3===0),borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.1)',fill:true,tension:.3,pointRadius:0}],{plugins:{legend:{display:false}}});

mc('c02','bar',D.eda.monthly_summary.months,[{label:'Legitimate',data:D.eda.monthly_summary.txn_count.map((t,i)=>t-D.eda.monthly_summary.fraud_count[i]),backgroundColor:'#3b82f6'},{label:'Fraud',data:D.eda.monthly_summary.fraud_count,backgroundColor:'#ef4444'}],{scales:{x:{stacked:true},y:{stacked:true}}});

const cd=D.eda.category_stats;
mc('c03','bar',cd.merchant_category,[{label:'Fraud Rate %',data:cd.fraud_rate,backgroundColor:'#f59e0b'}],{indexAxis:'y',plugins:{legend:{display:false}}});

const pm=D.eda.payment_method_stats;
mc('c04','bar',pm.payment_method,[{label:'Fraud Rate %',data:pm.fraud_rate,backgroundColor:'#8b5cf6'}],{plugins:{legend:{display:false}}});

document.getElementById('dq').innerHTML=`<table><tr><th>Metric</th><th>Value</th></tr><tr><td>Rows</td><td>${fmt(D.data_quality.shape[0])}</td></tr><tr><td>Columns</td><td>${D.data_quality.shape[1]}</td></tr><tr><td>Nulls</td><td>0</td></tr><tr><td>Duplicates</td><td>${D.data_quality.duplicates}</td></tr><tr><td>Amount P50</td><td>${gbp(D.data_quality.amount_percentiles.p50)}</td></tr><tr><td>Amount P99</td><td>${gbp(D.data_quality.amount_percentiles.p99)}</td></tr><tr><td>Customers</td><td>${fmt(D.data_quality.customer_id_unique)}</td></tr><tr><td>Merchants</td><td>${fmt(D.data_quality.merchant_id_unique)}</td></tr></table>`;

// TAB 1
mc('c11','bar',D.eda.hourly_fraud.hours,[{data:D.eda.hourly_fraud.rate,backgroundColor:D.eda.hourly_fraud.hours.map(h=>h<=5?'#ef4444':'#3b82f6')}],{plugins:{legend:{display:false}}});
mc('c12','bar',D.eda.dow_fraud.days,[{data:D.eda.dow_fraud.rate,backgroundColor:'#06b6d4'}],{plugins:{legend:{display:false}}});
mc('c13','bar',D.eda.amount_distribution.bins,[{label:'Legitimate',data:D.eda.amount_distribution.legit,backgroundColor:'#3b82f6'},{label:'Fraud',data:D.eda.amount_distribution.fraud,backgroundColor:'#ef4444'}],{scales:{y:{type:'logarithmic'}}});
mc('c14','line',D.eda.amount_distribution.bins,[{label:'Fraud Rate %',data:D.eda.amount_distribution.fraud_rate,borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,0.1)',fill:true,tension:.3}],{plugins:{legend:{display:false}}});

const coD=D.eda.country_stats;
mc('c15','bar',coD.country,[{data:coD.fraud_rate,backgroundColor:'#10b981'}],{plugins:{legend:{display:false}}});

const fac=D.eda.fraud_amount_by_cat,facK=Object.keys(fac);
mc('c16','bar',facK,[{label:'Legit Avg',data:facK.map(c=>fac[c].legit_mean),backgroundColor:'#3b82f6'},{label:'Fraud Avg',data:facK.map(c=>fac[c].fraud_mean),backgroundColor:'#ef4444'}]);

const tf=D.eda.top_fraud_customers;
document.getElementById('tf').innerHTML='<tr><th>Customer</th><th>Fraud Count</th><th>Fraud Value</th></tr>'+tf.customer_id.map((c,i)=>`<tr><td>${c}</td><td>${tf.fraud_count[i]}</td><td>${gbp(tf.fraud_value[i])}</td></tr>`).join('');

// TAB 2
const mdls=['LogisticRegression','RandomForest','HistGradientBoosting'],mC=['#3b82f6','#10b981','#f59e0b'],mS=['LogReg','RF','HistGBT'];
mkKPI('k2',mdls.map((m,i)=>({v:D.ml[m].test.roc_auc.toFixed(3),l:mS[i]+' ROC-AUC',c:mC[i],d:'PR-AUC: '+D.ml[m].test.pr_auc.toFixed(3)})));

new Chart('c21',{type:'scatter',data:{datasets:mdls.map((m,i)=>({label:mS[i]+' (AUC='+D.ml[m].test.roc_auc.toFixed(3)+')',data:D.ml[m].roc_curve.fpr.map((f,j)=>({x:f,y:D.ml[m].roc_curve.tpr[j]})),borderColor:mC[i],showLine:true,pointRadius:0})).concat([{label:'Random',data:[{x:0,y:0},{x:1,y:1}],borderColor:'#cbd5e1',borderDash:[5,5],showLine:true,pointRadius:0}])},options:{...JSON.parse(JSON.stringify(co)),scales:{x:{title:{display:true,text:'FPR',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}},y:{title:{display:true,text:'TPR',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}}}}});

new Chart('c22',{type:'scatter',data:{datasets:mdls.map((m,i)=>({label:mS[i]+' (AP='+D.ml[m].test.pr_auc.toFixed(3)+')',data:D.ml[m].pr_curve.recall.map((r,j)=>({x:r,y:D.ml[m].pr_curve.precision[j]})),borderColor:mC[i],showLine:true,pointRadius:0}))},options:{...JSON.parse(JSON.stringify(co)),scales:{x:{title:{display:true,text:'Recall',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}},y:{title:{display:true,text:'Precision',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}}}}});

const fi=D.ml.feature_importance||{},fiE=Object.entries(fi).slice(0,15).reverse();
mc('c23','bar',fiE.map(e=>e[0]),[{data:fiE.map(e=>e[1]),backgroundColor:'#06b6d4'}],{indexAxis:'y',plugins:{legend:{display:false}}});

const sd=D.explainability.score_distribution;
mc('c24','bar',sd.bins.map(b=>b.toFixed(2)),[{label:'Legitimate',data:sd.legit,backgroundColor:'rgba(59,130,246,0.6)'},{label:'Fraud',data:sd.fraud,backgroundColor:'rgba(239,68,68,0.6)'}],{scales:{y:{type:'logarithmic'}}});

const rn=Object.keys(D.rules);
document.getElementById('rt').innerHTML='<tr><th>Rule</th><th>Flagged</th><th>Prec</th><th>Recall</th><th>F1</th></tr>'+rn.map(r=>`<tr><td>${r}</td><td>${fmt(D.rules[r].flagged)}</td><td>${D.rules[r].precision.toFixed(3)}</td><td>${D.rules[r].recall.toFixed(3)}</td><td>${D.rules[r].f1.toFixed(3)}</td></tr>`).join('');

document.getElementById('mt').innerHTML='<tr><th>Model</th><th>Prec</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr>'+mdls.map(m=>{const t=D.ml[m].test;return`<tr><td>${m}</td><td>${t.precision.toFixed(3)}</td><td>${t.recall.toFixed(3)}</td><td>${t.f1.toFixed(3)}</td><td>${t.roc_auc.toFixed(3)}</td><td>${t.pr_auc.toFixed(3)}</td></tr>`;}).join('');

const fp=D.explainability.false_positive_analysis,fn=D.explainability.false_negative_analysis;
document.getElementById('fpp').innerHTML=`<table><tr><td>Count</td><td>${fmt(fp.count)}</td></tr><tr><td>Avg Amount</td><td>${gbp(fp.avg_amount)}</td></tr><tr><td>Avg Score</td><td>${fp.avg_prob.toFixed(4)}</td></tr><tr><td>Night %</td><td>${fp.night_pct}%</td></tr><tr><td>Cross-border %</td><td>${fp.cross_border_pct}%</td></tr><tr><td>Top Categories</td><td>${Object.entries(fp.by_category).sort((a,b)=>b[1]-a[1]).slice(0,3).map(e=>e[0]+': '+e[1]).join(', ')}</td></tr></table>`;
document.getElementById('fnp').innerHTML=`<table><tr><td>Missed Count</td><td>${fmt(fn.count)}</td></tr><tr><td>Missed Value</td><td>${gbp(fn.total_value)}</td></tr><tr><td>Avg Amount</td><td>${gbp(fn.avg_amount)}</td></tr><tr><td>Avg Score</td><td>${fn.avg_prob.toFixed(4)}</td></tr><tr><td>Top Categories</td><td>${Object.entries(fn.by_category).sort((a,b)=>b[1]-a[1]).slice(0,3).map(e=>e[0]+': '+e[1]).join(', ')}</td></tr></table>`;

// TAB 3
const bs=D.strategy.strategies.balanced;
mkKPI('k3',[{v:gbp(bs.cost.total),l:'Total Cost (Balanced)',c:'#10b981'},{v:bs.fraud.value_prevented_pct+'%',l:'Fraud Prevented',c:'#3b82f6'},{v:bs.legitimate.approval_rate+'%',l:'Legit Approval Rate',c:'#06b6d4'},{v:bs.decisions.reviewed_pct+'%',l:'Review Rate',c:'#f59e0b'}]);

const sn=Object.keys(D.strategy.strategies);
mc('c31','bar',sn.map(s=>s[0].toUpperCase()+s.slice(1)),[{label:'Fraud Losses',data:sn.map(s=>D.strategy.strategies[s].cost.fraud_losses),backgroundColor:'#ef4444'},{label:'False Decline',data:sn.map(s=>D.strategy.strategies[s].cost.false_decline_cost),backgroundColor:'#f59e0b'},{label:'Review Cost',data:sn.map(s=>D.strategy.strategies[s].cost.review_cost),backgroundColor:'#3b82f6'}],{scales:{x:{stacked:true},y:{stacked:true}}});

new Chart('c32',{type:'scatter',data:{datasets:sn.map((s,i)=>({label:s[0].toUpperCase()+s.slice(1),data:[{x:D.strategy.strategies[s].fraud.value_prevented_pct,y:D.strategy.strategies[s].legitimate.approval_rate}],backgroundColor:['#ef4444','#10b981','#3b82f6','#f59e0b'][i],pointRadius:10}))},options:{...JSON.parse(JSON.stringify(co)),scales:{x:{title:{display:true,text:'Fraud Prevented %',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}},y:{title:{display:true,text:'Legit Approval %',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}}}}});

document.getElementById('st').innerHTML='<tr><th>Strategy</th><th>Approve%</th><th>Review%</th><th>Decline%</th><th>Fraud Prevented</th><th>Legit Approval</th><th>Cost</th></tr>'+sn.map(s=>{const x=D.strategy.strategies[s];return`<tr><td>${s}</td><td>${x.decisions.approved_pct}%</td><td>${x.decisions.reviewed_pct}%</td><td>${x.decisions.declined_pct}%</td><td>${x.fraud.value_prevented_pct}%</td><td>${x.legitimate.approval_rate}%</td><td>${gbp(x.cost.total)}</td></tr>`;}).join('');

mc('c33','bar',sn.map(s=>s[0].toUpperCase()+s.slice(1)),[{label:'Fraud Losses',data:sn.map(s=>D.strategy.strategies[s].cost.fraud_losses),backgroundColor:'#ef4444'},{label:'False Declines',data:sn.map(s=>D.strategy.strategies[s].cost.false_decline_cost),backgroundColor:'#f59e0b'},{label:'Review',data:sn.map(s=>D.strategy.strategies[s].cost.review_cost),backgroundColor:'#3b82f6'}],{indexAxis:'y',scales:{x:{stacked:true},y:{stacked:true}}});

const dr=D.explainability.monthly_drift;
new Chart('c34',{type:'line',data:{labels:dr.map(d=>'Month '+d.month),datasets:[{label:'Fraud Rate %',data:dr.map(d=>(d.fraud_rate*100).toFixed(2)),borderColor:'#ef4444',yAxisID:'y'},{label:'Avg Score %',data:dr.map(d=>(d.avg_score*100).toFixed(2)),borderColor:'#3b82f6',yAxisID:'y1'}]},options:{responsive:true,plugins:{legend:{labels:{color:'#475569'}}},scales:{x:{ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}},y:{position:'left',title:{display:true,text:'Fraud Rate %',color:'#ef4444'},ticks:{color:'#ef4444'},grid:{color:'#e2e8f0'}},y1:{position:'right',title:{display:true,text:'Avg Score %',color:'#3b82f6'},ticks:{color:'#3b82f6'},grid:{display:false}}}}});

// TAB 4
const swp=D.strategy.threshold_sweep;
mc('c41','line',swp.map(s=>s.threshold),[{label:'Total Cost',data:swp.map(s=>s.total_cost),borderColor:'#10b981',pointRadius:0,tension:.2}],{plugins:{legend:{display:false}},scales:{x:{title:{display:true,text:'Threshold',color:'#475569'}}}});
mc('c42','line',swp.map(s=>s.threshold),[{label:'Precision',data:swp.map(s=>s.precision),borderColor:'#3b82f6',pointRadius:0},{label:'Recall',data:swp.map(s=>s.recall),borderColor:'#ef4444',pointRadius:0},{label:'F1',data:swp.map(s=>s.f1),borderColor:'#f59e0b',pointRadius:0}],{scales:{x:{title:{display:true,text:'Threshold',color:'#475569'}}}});

let sdn=new Chart('c43',{type:'doughnut',data:{labels:['Approved','Review','Declined'],datasets:[{data:[90,8,2],backgroundColor:['#10b981','#f59e0b','#ef4444']}]},options:{responsive:true,plugins:{legend:{labels:{color:'#475569'}}}}});

new Chart('c44',{type:'line',data:{labels:swp.map(s=>s.threshold),datasets:[{label:'Fraud Captured %',data:swp.map(s=>s.fraud_value_captured_pct),borderColor:'#10b981',pointRadius:0,yAxisID:'y'},{label:'FPR',data:swp.map(s=>s.fpr),borderColor:'#ef4444',pointRadius:0,yAxisID:'y1'}]},options:{responsive:true,plugins:{legend:{labels:{color:'#475569'}}},scales:{x:{title:{display:true,text:'Threshold',color:'#475569'},ticks:{color:'#64748b'},grid:{color:'#e2e8f0'}},y:{position:'left',title:{display:true,text:'Fraud Captured %',color:'#10b981'},ticks:{color:'#10b981'},grid:{color:'#e2e8f0'}},y1:{position:'right',title:{display:true,text:'FPR',color:'#ef4444'},ticks:{color:'#ef4444'},grid:{display:false}}}}});

function usim(){
const tr=parseFloat(document.getElementById('sr').value);
let td=parseFloat(document.getElementById('sd').value);
if(td<=tr){td=tr+0.01;document.getElementById('sd').value=td;}
document.getElementById('vr').textContent=tr.toFixed(2);
document.getElementById('vd').textContent=td.toFixed(2);
const cl=swp.reduce((b,s)=>Math.abs(s.threshold-td)<Math.abs(b.threshold-td)?s:b);
document.getElementById('sk').innerHTML=`<div class="c kpi"><div class="v" style="color:#10b981">${gbp(cl.total_cost)}</div><div class="l">Est. Cost</div></div><div class="c kpi"><div class="v" style="color:#3b82f6">${cl.fraud_value_captured_pct}%</div><div class="l">Fraud Captured</div></div><div class="c kpi"><div class="v" style="color:#06b6d4">${(cl.legit_approval_rate*100).toFixed(1)}%</div><div class="l">Legit Approval</div></div><div class="c kpi"><div class="v" style="color:#f59e0b">${cl.precision.toFixed(3)}</div><div class="l">Precision</div></div>`;
// Use both thresholds: find closest sweep entry for review threshold too
const clr=swp.reduce((b,s)=>Math.abs(s.threshold-tr)<Math.abs(b.threshold-tr)?s:b);
const approved=clr.tn+clr.fn; // below review threshold: legit passed + fraud missed
const total=cl.tp+cl.fp+cl.tn+cl.fn;
const declined=cl.tp+cl.fp; // above decline threshold: blocked
const reviewed=Math.max(0,total-approved-declined); // between thresholds
sdn.data.datasets[0].data=[approved,reviewed,declined];sdn.update();}
usim();
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Payments Fraud & Risk Decisioning Engine</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>{CSS}</style>
</head>
<body>
{BODY}
<script>
const D = {dj};
{JS}
</script>
</body>
</html>"""

out = f"{ROOT}/dashboard/dashboard.html"
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Dashboard: {os.path.getsize(out):,} bytes -> {out}")
