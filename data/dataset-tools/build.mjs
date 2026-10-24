import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';

const output = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const features = ['packet_rate','failed_logins','unique_ports','avg_packet_size','connection_duration','syn_ratio'];
const centers = {
  benign:[125,1,4,760,32,.20], reconnaissance:[230,1,35,450,8,.45],
  credential_access:[210,12,7,560,14,.35], lateral_movement:[310,5,22,680,26,.40],
  exfiltration:[430,1,6,1250,75,.17], denial_of_service:[780,2,12,230,3,.90],
};
function random(seed) {
  let state=seed>>>0;
  return ()=> { state+=0x6D2B79F5; let t=state; t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296; };
}
function sample(stage,rng,noise=1) {
  const spread=[65,2.5,5,160,9,.10];
  return centers[stage].map((mean,index)=> {
    const z=Math.sqrt(-2*Math.log(Math.max(rng(),1e-12)))*Math.cos(2*Math.PI*rng());
    let value=Math.max(0,mean+spread[index]*noise*z);
    if(index===5)value=Math.min(1,value);
    if(index===1||index===2)value=Math.round(value);
    return Number(value.toFixed(4));
  });
}
const paths = [
  ['benign','benign','benign'],
  ['reconnaissance','credential_access','lateral_movement','exfiltration'],
  ['reconnaissance','denial_of_service','denial_of_service'],
  ['credential_access','credential_access','benign'],
  ['reconnaissance','reconnaissance','benign'],
  ['credential_access','lateral_movement','exfiltration'],
];
function sequences({episodes,seed,prefix,start,variable=false,noise=1}) {
  const rng=random(seed);const rows=[];
  for(let episode=0;episode<episodes;episode++) {
    const stages=paths[episode%paths.length];let seconds=0;
    for(const stage of stages) {
      const count=variable ? 2+Math.floor(rng()*5) : 3;
      for(let i=0;i<count;i++) {
        const timestamp=new Date(Date.parse(start)+episode*3600000+seconds*1000).toISOString();
        rows.push([`${prefix}-${String(episode).padStart(4,'0')}`,timestamp,stage,'synthetic',...sample(stage,rng,noise)]);
        seconds+=variable ? 10+Math.floor(rng()*81) : 30;
      }
    }
  }
  return rows;
}
const specs=[
  {name:'synthetic_sequence_baseline.csv',episodes:240,seed:90311,prefix:'baseline',start:'2025-02-01T00:00:00Z',purpose:'Baseline sequence training with fixed stage dwell lengths.'},
  {name:'synthetic_sequence_variable.csv',episodes:360,seed:90312,prefix:'variable',start:'2025-03-01T00:00:00Z',variable:true,noise:1.2,purpose:'Sequence training with variable dwell lengths, timing and more feature overlap.'},
  {name:'synthetic_sequence_evaluation.csv',episodes:120,seed:90313,prefix:'evaluation',start:'2025-05-01T00:00:00Z',variable:true,noise:1.2,purpose:'Reserved later evaluation; do not upload to training.'},
];
const datasets=specs.map(spec=>({...spec,headers:['episode_id','timestamp','attack_stage','data_source',...features],rows:sequences(spec)}));
const rng=random(90314);const attacks=Object.keys(centers).filter(x=>x!=='benign');
const binary=[];
for(let i=0;i<6000;i++){const label=i%2;binary.push([...sample(label?attacks[Math.floor(rng()*attacks.length)]:'benign',rng,1.1),label,'synthetic']);}
datasets.push({name:'synthetic_binary_balanced.csv',seed:90314,purpose:'Balanced binary IDS training; 3000 normal and 3000 generated attack examples.',headers:[...features,'label','data_source'],rows:binary});

const manifest=[];
for(const dataset of datasets) {
  const wb=Workbook.create();const sheet=wb.worksheets.add('Data');
  const matrix=[dataset.headers,...dataset.rows];const range=sheet.getRangeByIndexes(0,0,matrix.length,dataset.headers.length);
  if(dataset.headers.includes('timestamp'))sheet.getRangeByIndexes(0,1,matrix.length,1).setNumberFormat('@');
  range.values=matrix;
  wb.recalculate();
  const values=range.values.map(row=>row.map(value=>value instanceof Date ? value.toISOString() : value));
  if(values.length!==matrix.length||values[1].length!==dataset.headers.length)throw Error('CSV shape changed');
  for(let row=0;row<matrix.length;row++)for(let col=0;col<dataset.headers.length;col++)if(values[row][col]!==matrix[row][col])throw Error(`Value changed at ${row},${col}: ${JSON.stringify(values[row][col])} expected ${JSON.stringify(matrix[row][col])}`);
  const quote=value=>/[",\r\n]/.test(String(value))?`"${String(value).replaceAll('"','""')}"`:String(value);
  const csv=values.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n';
  await fs.writeFile(path.join(output,dataset.name),csv,'utf8');
  // Render only a representative raw-data slice; CSV itself has no visual formatting.
  const previewRange=sheet.getRangeByIndexes(0,0,6,dataset.headers.length);
  previewRange.format.columnWidth=24;
  previewRange.format.rowHeight=34;
  previewRange.format.wrapText=true;
  sheet.getRangeByIndexes(0,0,1,dataset.headers.length).format.font={bold:true};
  const preview=await wb.render({sheetName:'Data',range:`A1:${dataset.headers.length===10?'J':'H'}6`,scale:1,format:'png'});
  await fs.writeFile(path.join(output,'dataset-tools',dataset.name+'.png'),new Uint8Array(await preview.arrayBuffer()));
  const labelIndex=dataset.headers.indexOf('attack_stage')>=0?dataset.headers.indexOf('attack_stage'):dataset.headers.indexOf('label');
  const counts={};for(const row of dataset.rows)counts[row[labelIndex]]=(counts[row[labelIndex]]||0)+1;
  manifest.push({file:dataset.name,rows:dataset.rows.length,episodes:dataset.episodes||null,seed:dataset.seed,purpose:dataset.purpose,classes:counts,sha256:createHash('sha256').update(csv).digest('hex')});
}
await fs.writeFile(path.join(output,'dataset_manifest.json'),JSON.stringify({source:'Synthetic; no real traffic captured',generator:'Mulberry32, Box-Muller normal draws; illustrative centers inherited from project generator',datasets:manifest},null,2));
console.log(JSON.stringify(manifest,null,2));
