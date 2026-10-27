(function(){'use strict';
const root=document.getElementById('analysis');if(!root)return;
const node=id=>document.getElementById(id),periods=[...root.querySelectorAll('button[data-period]')];
const fmt=n=>new Intl.NumberFormat('en-US',{maximumFractionDigits:4}).format(n);
const eur=n=>new Intl.NumberFormat('en-IE',{style:'currency',currency:'EUR'}).format(n);
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39'}[c]));
const dayLabel=d=>{try{return new Date(d+'T12:00:00Z').toLocaleDateString('en-GB',{day:'numeric',month:'short'})}catch{return d}};
/*
 * Data correction (presentation only, full historical transactions untouched).
 * Independent sync on 22 Oct 2026 already observed 4.6365 points, but the
 * existing ledger labeled it 23 Oct. The 23 Oct scan added 4.6394 more.
 * The two entries were combined under 23 Oct (9.2759), leaving 22 Oct blank.
 * Only correct this exact independently-verified anomaly; never estimate other days.
 */
function correctLedgerDateAnomaly(result) {
 if(!result || result.dateBasis!=='credited-next-morning' ||
    !Array.isArray(result.chart)) return result;
 const oct22=result.chart.find(x=>x.day==='2026-10-22');
 const oct23=result.chart.find(x=>x.day==='2026-10-23');
 const amount=4.6365;
 const matches=oct22 && oct23 &&
   Math.abs(Number(oct22.points||0))<0.00005 &&
   Math.abs(Number(oct23.points||0)-9.2759)<0.00005 &&
   Number(result.archiveTransactions)>=86;
 if(!matches) return result;
 const copy={...result,chart:result.chart.map(x=>{
   if(x.day==='2026-10-22')return {...x,points:amount};
   if(x.day==='2026-10-23')return {...x,points:Math.round((Number(x.points)-amount)*10000)/10000};
   return x;
 }),_dateDisplayCorrection:true};
 if(Array.isArray(result.recentDays)){
   const rows=result.recentDays.filter(x=>x.day!=='2026-10-22'&&x.day!=='2026-10-23');
   rows.push({day:'2026-10-22',points:amount},{day:'2026-10-23',points:4.6394});
   copy.recentDays=rows.sort((a,b)=>b.day.localeCompare(a.day));
 }
 return copy;
}


/* Live-sync truthfulness: only authenticated CurseForge rewards checks verify balance.
   Public mod downloads cannot be converted to reward points. */
const greeceDay = date => {
 try {
   const x=new Intl.DateTimeFormat('en-GB',{year:'numeric',month:'2-digit',day:'2-digit',timeZone:'Europe/Athens'}).formatToParts(new Date(date));
   const g=k=>x.find(y=>y.type===k)?.value||'';
   return g('year')+'-'+g('month')+'-'+g('day');
 } catch { return ''; }
};
const syncStamp = date => {
 try { return new Date(date).toLocaleString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'Europe/Athens'}); }
 catch { return 'unknown'; }
};
function markUnsyncedToday(data) {
 const notToday=!data.lastSuccessAt||greeceDay(data.lastSuccessAt)!==greeceDay(new Date());
 const blocked=/browserbase|browser minutes|\b402\b|quota|limit reached/i.test(String(data.lastError||''));
 if(period!=='today'||Number(data.transactions||0)!==0||(!notToday&&!blocked))return;
 const badge=node('analysisBadge'),total=node('analysisTotal'),count=node('analysisCount'),board=node('analysisBoard');
 if(badge)badge.textContent='⚠ Not synced today';
 if(total)total.textContent='—';
 if(count)count.textContent='Awaiting verified import';
 if(board)board.innerHTML='<div class="analysisEmpty"><b>Today: no verified data (not zero points).</b>The last completed rewards import was '+escape(syncStamp(data.lastSuccessAt))+'. A blocked browser session prevents a reliable new total. Previously imported points remain saved.</div>';
 const chart=node('analysisGraph');
 if(chart&&Array.isArray(data.chart)){
   const days=Array.from(chart.querySelectorAll('.analysisDay'));
   data.chart.forEach((point,i)=>{
     if(!point||point.day!==greeceDay(new Date())||Number(point.points||0)!==0||!days[i])return;
     const bar=days[i].querySelector('.analysisColumn'),tick=days[i].querySelector('small');
     if(bar){bar.title='Not synced / data unavailable; this is not zero earned points';bar.style.height='0px';}
     if(tick)tick.textContent=point.day.slice(-2)+'?';
   });
 }
}
async function showSyncHealth() {
 let details;
 try {
   const [br,dr]=await Promise.all([
     fetch('/api/browser?action=status',{cache:'no-store',credentials:'same-origin'}),
     fetch('/api/data',{cache:'no-store',credentials:'same-origin'})
   ]);
   if(!br.ok||!dr.ok)throw Error('sync status API unavailable');
   const browser=await br.json(),data=await dr.json();
   const last=data.updatedAt||null;
   const stale=!last||greeceDay(last)!==greeceDay(new Date());
   const quota=/\b402\b|browser minutes limit reached/i.test(String(browser.lastError||''));
   details={
     failed:!browser.connected||stale,
     message:quota?'SYNC PAUSED — browser minutes exhausted':(!browser.connected?'SYNC DISCONNECTED':(stale?'POINTS NOT UPDATED TODAY':'LAST POINTS CHECK COMPLETE')),
     explanatory:quota?'CurseForge reward balance cannot be checked because the browser service has hit its free minutes limit.':(!browser.connected?'CurseForge browser connection unavailable.':'The dashboard shows the last successfully saved reading, not a live balance.'),
     last:last?syncStamp(last):'never',
     points:Number.isFinite(Number(data.points))?String(data.points):'unknown'
   };
 }catch{details={failed:true,message:'SYNC STATUS UNKNOWN',explanatory:'Could not confirm a fresh authenticated rewards check.',last:'unavailable',points:'unknown'};}
 const pill=document.querySelector('.syncStatus');
 if(pill){
   pill.textContent=(details.failed?'⚠ ':'● ')+details.message;
   pill.title=details.explanatory+' Last saved: '+details.last+'.';
   if(details.failed){pill.style.background='#4a2922';pill.style.color='#ffd0ab';pill.style.borderColor='#98674b';}
   else {pill.style.background='';pill.style.color='';pill.style.borderColor='';}
 }
 let box=document.getElementById('foxGoalsRewardsHealthNotice');
 if(!box){
   const target=document.querySelector('.heroControls')||root;
   box=document.createElement('div');box.id='foxGoalsRewardsHealthNotice';box.setAttribute('role','status');
   box.style.cssText='margin:12px 0;padding:12px 15px;border-radius:12px;border:1px solid #825d45;background:#30251e;color:#ffe1c3;font-size:12px;line-height:1.7;';
   if(target?.parentNode)target.parentNode.insertBefore(box,target.nextSibling);
 }
 if(box){
   box.textContent=details.failed
     ?details.message+'. '+details.explanatory+' Last saved: '+details.points+' points at '+details.last+'. Missing days are unknown, not zero.'
     :'Latest recorded authenticated check: '+details.points+' points at '+details.last+'.';
   box.style.display=details.failed?'block':'none';
 }
}
showSyncHealth();
setInterval(showSyncHealth,60000);
window.addEventListener('foxgoals:refresh',()=>showSyncHealth());

let period='today',busy=false,seenSignature='',refreshQueued=false;
const oneDecimal=n=>new Intl.NumberFormat('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}).format(n);
const readableDay=s=>s?dayLabel(s):'not available';
periods.forEach(b=>{const active=b.dataset.period===period;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
periods.forEach(b=>b.addEventListener('click',()=>{if(b.dataset.period===period)return;period=b.dataset.period;periods.forEach(p=>{let yes=p.dataset.period===period;p.classList.toggle('active',yes);p.setAttribute('aria-pressed',String(yes))});refresh(true)}));

function renderPace(pace){
 const p=pace||{},week=p.recent7||{},month=p.month||{},all=p.allTime||{};
 function fill(id,meta,extra,metric,description){
  const available=metric.available&&Number.isFinite(metric.pointsPerDay);
  node(id).textContent=available?oneDecimal(metric.pointsPerDay)+' pts/day':'—';
  node(meta).textContent=available?fmt(metric.points)+' points across '+metric.days+' calendar days':'Not enough complete history to calculate';
  if(extra)node(extra).textContent=available?description(metric):'Available once more transactions are recorded';
 }
 fill('velocity7','velocity7meta','velocityProjection',week,m=>((Number.isFinite(p.estimated30DayEUR)?'≈ '+eur(p.estimated30DayEUR)+' / 30 days at this pace':'')+(Number.isFinite(p.trendPercent)?' · '+(p.trendPercent>=0?'+':'')+oneDecimal(p.trendPercent)+'% vs previous 7 days':'')));
 fill('velocityMonth','velocityMonthMeta','velocityMonthExtra',month,m=>'Month-to-date average · '+m.from+' to '+m.through);
 fill('velocityAll','velocityAllMeta','velocityAllExtra',all,m=>'Full history '+readableDay(m.from)+' to '+readableDay(m.through));
 let explanation='Actual points generated ÷ calendar days, including zero-point days. 7-day pace is the most useful recent guide; this is not a guarantee of future income.';
 if(p.asOf)explanation+=' Latest recorded day: '+readableDay(p.asOf)+'.';
 if(p.stale)explanation+=' Data is '+p.daysBehind+' days behind today; estimates may be stale.';
 if(week.includesScreenshot||month.includesScreenshot||all.includesScreenshot)explanation+=' Some values include a user-provided screenshot entry.';
 if(!p.coverageComplete)explanation+=' All-time average is unavailable until the history is complete.';
 node('velocityContext').textContent=explanation;
}

function renderChart(days){const chart=node('analysisGraph');chart.innerHTML='';const maximum=Math.max(0.02,...days.map(d=>d.points));for(const d of days){
 const col=document.createElement('div');col.className='analysisDay';
 const bar=document.createElement('span');bar.className='analysisColumn';bar.style.height=Math.max(2,Math.round(d.points/maximum*130))+'px';bar.title=d.day+' · '+fmt(d.points)+' points';
 const tick=document.createElement('small');tick.textContent=d.day.slice(8);
 col.append(bar,tick);chart.append(col);
}}
function board(data){
 const items=data.projects||[];
 if(!items.length&&period==='today'&&data.coveredThrough){
  node('analysisBoard').innerHTML='<div class="analysisEmpty"><b>No verified points released today yet</b>Last verified transaction date: '+escape(dayLabel(data.coveredThrough))+'. The latest authenticated release is not yet available for today. See previous days under Last 7 days.</div>';
  return
 }
 if(!items.length){node('analysisBoard').innerHTML=(Number(data.unattributed)>0?'<div class="analysisEmpty"><b>Project breakdown is still being imported</b>The generated points are verified, but per-project allocations are not yet confirmed. Rankings will fill in after the next successful daily rewards check.</div>':'<div class="analysisEmpty"><b>No project earnings recorded for this period</b>New transactions will appear here after the daily check.</div>');return}
 const top=Math.max(...items.map(p=>p.points),0.001);
 node('analysisBoard').innerHTML=items.map((p,i)=>'<div class="analysisRank"><span class="analysisMedal">'+(i===0?'🥇':i===1?'🥈':i===2?'🥉':'#'+(i+1))+'</span><div class="analysisRankInfo"><div class="analysisRankName">'+escape(p.name)+'</div><div class="analysisRankMeta">'+((data.attributed>0)?(100*p.points/data.attributed).toFixed(1)+'% of attributed points':'Project earnings')+'</div><div class="analysisMiniBar"><span style="width:'+Math.max(0,Math.min(100,p.points/top*100)).toFixed(2)+'%"></span></div></div><div class="analysisRankValue"><strong>'+fmt(p.points)+' pts</strong><span>'+eur(p.points*(data.total>0?data.estimatedEUR/data.total:.044685))+' est.</span></div></div>').join('');
}
function render(data){
 if(!data.ready){
  node('analysisTotal').textContent='—';node('analysisEuro').textContent='—';node('analysisTop').textContent='—';
  node('analysisBoard').innerHTML='<div class="analysisEmpty"><b>Awaiting first transaction import</b>Next scheduled points check: approximately 9:30 AM Greece time. No extra browser session will be used.</div>';
  node('analysisInfo').innerHTML='<strong>Not available yet.</strong> '+escape(data.lastError||data.message||'The first verified transaction import has not run.');
  node('analysisBadge').textContent='◷ Awaiting first sync';
  node('analysisCount').textContent='No transactions';
  node('analysisDays').innerHTML='';node('analysisGraph').innerHTML='';return
 }
 node('analysisTotal').textContent=fmt(data.total);node('analysisEuro').textContent=eur(data.estimatedEUR);
 node('analysisTop').textContent=data.projects[0]?.name||'—';
 node('analysisTopDetail').textContent=data.projects[0]?fmt(data.projects[0].points)+' points earned':'No attributed projects in this period';
 node('analysisBadge').textContent='● Synced: '+new Date(data.lastSuccessAt).toLocaleDateString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'Europe/Athens'})+(data.coveredThrough?' · latest credited '+dayLabel(data.coveredThrough):'');
 node('analysisCount').textContent=fmt(data.transactions)+' transactions';
 board(data);renderChart(data.chart||[]);
 node('analysisDays').innerHTML=(data.recentDays||[]).slice(0,8).map(x=>'<div class="analysisDailyRow"><span>'+escape(dayLabel(x.day))+'</span><b>'+fmt(x.points)+' pts</b></div>').join('')||'<p class="analysisSub">No point-generation transactions in this period.</p>';
 let text='<strong>Verified rewards data'+(data.manualSnapshotsIncluded?' + screenshot entry':'')+'.</strong> '+fmt(data.archiveTransactions)+' verified transactions saved · last import '+new Date(data.lastSuccessAt).toLocaleString('en-GB',{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit',timeZone:'Europe/Athens'})+'.';
 if(data.coveredThrough){text+=' <strong>Latest credited earnings:</strong> '+escape(dayLabel(data.coveredThrough))+'.';}
 if(data.dateBasis==='credited-next-morning'&&data.latestLedgerDay){
  text+=' <strong>Crediting schedule:</strong> '+escape(dayLabel(data.latestLedgerDay))+' generation was credited on '+escape(dayLabel(data.coveredThrough))+'.';
 }
 if(period==='today'&&data.transactions===0&&data.coveredThrough){text+=' <strong>Today has no newly posted ledger entries yet.</strong> The points balance and daily transaction ledger can update on different schedules. Select Last 7 days for the newest confirmed earnings.';}
 if(data.manualSnapshotsIncluded)text+=' <strong>Screenshot-based entry:</strong> '+(data.manualSnapshotDates||[]).map(escape).join(', ')+' — '+fmt(data.manualSnapshotPoints)+' points entered from the provided screenshot (not independently verified). It will be replaced automatically once a matching dated CurseForge transaction is imported.';
 if(!data.coverageComplete)text+=' <strong>History is still being collected:</strong> all-time totals may be incomplete.';
 if(data.unattributed>0.0001)text+=' '+fmt(data.unattributed)+' points are unattributed because their project breakdown was unavailable or rounded.';
 if(data.warning)text+=' '+escape(data.warning);
 if(data._dateDisplayCorrection)text+=' <strong>Verified historical correction:</strong> A 4.6365-point entry first seen on 22 Oct was incorrectly grouped with a 4.6394-point entry on 23 Oct. The graph and daily rows now show the two dates separately. Raw stored transactions remain unchanged, and no points were invented.';
 if(data.lastError)text+=' <strong>Latest import issue:</strong> '+escape(data.lastError);
 node('analysisInfo').innerHTML=text;
}
async function refresh(force=false){
 if(busy){if(force)refreshQueued=true;return;}
 if(!force&&document.hidden)return;
 busy=true;
 try{
 const r=await fetch('/api/analysis?period='+encodeURIComponent(period)+'&t='+Date.now(),{cache:'no-store',credentials:'same-origin'});
 if(!r.ok)throw Error('The private analysis service returned HTTP '+r.status);
 const data=correctLedgerDateAnomaly(await r.json()),sig=JSON.stringify({period,at:data.lastSuccessAt,ready:data.ready,error:data.lastError,total:data.total,manual:data.manualSnapshotsIncluded,dates:data.manualSnapshotDates,pace:data.pace});
 if(data.pace){renderPace(data.pace);window.dispatchEvent(new CustomEvent('foxgoals:pace',{detail:data.pace}));}
 if(force||sig!==seenSignature){render(data);seenSignature=sig}
 markUnsyncedToday(data);
 }catch(e){node('analysisInfo').textContent='The points analysis is temporarily unavailable. '+String(e.message||e)}
 finally{busy=false;if(refreshQueued){refreshQueued=false;Promise.resolve().then(()=>refresh(true));}}
}
document.addEventListener('click',e=>{if(e.target.closest('[data-tab="analysis"]'))refresh(true)});
window.addEventListener('foxgoals:refresh',()=>refresh(true));
window.addEventListener('foxgoals:rateChanged',()=>refresh(true));
refresh(true);setInterval(()=>refresh(false),60000);
})();