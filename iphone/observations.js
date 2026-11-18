/* Descriptive camera features. No force labels, personal calibration, or kgf inference. */
(function(root){
  'use strict';
  const C=root.PrexCore,VERSION='iphone_observation_v2';
  const PHASES=['rest','squeeze','release'];
  const LABELS={rest:'기준 파지',squeeze:'최대 노력 파지',release:'이완 및 회복'};
  const CHANNELS={hand:'hand_flexion_change_mean_deg',observed:'observed_width_reduction_pct',proxy:'proxy_width_reduction_pct'};
  const {finite,quantile,mean}=C;
  function signal(record,name,relative=true){
    const values=record.rows.map(r=>finite(r.f[name])?r.f[name]:null);
    if(!relative)return values;
    const rest=values.filter((v,i)=>record.rows[i].phase==='rest'&&finite(v));
    if(rest.length<3)return values.map(()=>null);
    const base=quantile(rest),width=name.endsWith('width_reduction_pct')||name==='observed_area_reduction_pct';
    if(width&&100-base<=1e-6)return values.map(()=>null);
    return values.map(v=>!finite(v)?null:width?100*(v-base)/(100-base):v-base);
  }
  function solve3(a,b){
    const m=a.map((r,i)=>[...r,b[i]]);
    for(let i=0;i<3;i++){
      let k=i;for(let j=i+1;j<3;j++)if(Math.abs(m[j][i])>Math.abs(m[k][i]))k=j;
      [m[i],m[k]]=[m[k],m[i]];if(Math.abs(m[i][i])<1e-12)return null;
      const d=m[i][i];for(let j=i;j<4;j++)m[i][j]/=d;
      for(let r=0;r<3;r++)if(r!==i){const v=m[r][i];for(let j=i;j<4;j++)m[r][j]-=v*m[i][j];}
    }return m.map(r=>r[3]);
  }
  function rates(times,values){
    const speed=values.map(()=>null),acceleration=values.map(()=>null);let start=0;
    while(start<values.length){
      if(!finite(values[start])){start++;continue;}
      let end=start+1;
      while(end<values.length&&finite(values[end])&&times[end]>times[end-1]&&times[end]-times[end-1]<=.35)end++;
      for(let i=start;i<end;i++){
        const lo=Math.max(start,i-2),hi=Math.min(end,i+3),t=times.slice(lo,hi).map(x=>x-times[i]),v=values.slice(lo,hi);
        if(t.length>=3){const tm=mean(t),vm=mean(v),den=t.reduce((s,x)=>s+(x-tm)**2,0);
          if(den>1e-12)speed[i]=t.reduce((s,x,k)=>s+(x-tm)*(v[k]-vm),0)/den;}
        if(t.length>=5){const s=[0,1,2,3,4].map(k=>t.reduce((z,x)=>z+x**k,0));
          const fit=solve3([[s[0],s[1],s[2]],[s[1],s[2],s[3]],[s[2],s[3],s[4]]],
            [0,1,2].map(k=>t.reduce((z,x,j)=>z+x**k*v[j],0)));
          if(fit)acceleration[i]=2*fit[2];}
      }start=end;
    }return {speed,acceleration};
  }
  function describe(record,name){
    const values=signal(record,name),times=record.rows.map(r=>r.t),r=rates(times,values);
    const phases=Object.fromEntries(PHASES.map(p=>{
      const indexes=record.rows.flatMap((row,i)=>row.phase===p?[i]:[]),v=indexes.map(i=>values[i]).filter(finite);
      return [p,{valid:v.length,total:indexes.length,coverage_pct:indexes.length?100*v.length/indexes.length:0,
        median:quantile(v),p10:quantile(v,.1),p90:quantile(v,.9),min:v.length?Math.min(...v):null,max:v.length?Math.max(...v):null}];
    }));
    const sq=record.rows.flatMap((row,i)=>row.phase==='squeeze'?[i]:[]),rest=record.rows.flatMap((row,i)=>row.phase==='rest'&&finite(values[i])?[values[i]]:[]);
    const noise=rest.length>=3?1.4826*quantile(rest.map(v=>Math.abs(v-quantile(rest)))):null;
    const speeds=sq.map(i=>r.speed[i]).filter(finite),acc=sq.map(i=>r.acceleration[i]).filter(finite);
    const baseline=phases.rest.median,peak=phases.squeeze.p90,release=phases.release.median;
    const net=finite(peak)&&finite(baseline)?peak-baseline:null;
    const gap=record.rows.some((row,i)=>i&&row.phase==='squeeze'&&record.rows[i-1].phase==='squeeze'&&row.t-record.rows[i-1].t>.75);
    const supported=phases.rest.valid>=4&&phases.squeeze.valid>=8&&phases.release.valid>=4&&phases.squeeze.coverage_pct>=70&&!gap&&record.completed;
    return {phases,p90:peak,net_change:net,rest_noise:noise,supported,
      range:finite(peak)&&finite(phases.squeeze.p10)?peak-phases.squeeze.p10:null,
      speed_abs_p90:quantile(speeds.map(Math.abs),.9),speed_positive_p90:quantile(speeds.filter(v=>v>0),.9),
      acceleration_abs_p90:quantile(acc.map(Math.abs),.9),release_residual:release,
      signal_to_noise:finite(net)&&noise>1e-9?net/noise:null,
      return_pct:net>1e-6&&finite(release)?100*(peak-release)/net:null};
  }
  function buildRecord(input,meta,completed=true){
    const checked=C.validateMetadata(meta);
    if(typeof meta.created_at!=='string'||!Number.isFinite(Date.parse(meta.created_at)))throw Error('측정 날짜가 올바르지 않습니다.');
    const seen=new Set(),rows=[];
    for(const row of input){
      if(!row||!finite(row.t)||row.t<0||row.t>=7||!Number.isSafeInteger(row.id)||seen.has(row.id))continue;
      seen.add(row.id);const f=Object.fromEntries([...C.SIGNALS,'valid_hand','valid_axis','valid_observed','valid_proxy'].map(n=>[n,finite(row.f?.[n])?row.f[n]:null]));
      for(const flag of ['valid_hand','valid_axis','valid_observed','valid_proxy'])f[flag]=Number(f[flag]===1);
      if(!f.valid_axis){f.valid_observed=0;f.valid_proxy=0;}
      for(const name of C.SIGNALS){
        if(name.startsWith('observed_')?!f.valid_observed:name.startsWith('proxy_')?!f.valid_proxy:!f.valid_hand)f[name]=null;
      }
      rows.push({id:row.id,t:row.t,phase:C.phase(row.t),f,processing_ms:finite(row.processing_ms)?row.processing_ms:null});
    }
    rows.sort((a,b)=>a.t-b.t);
    const record={schema_version:2,feature_version:VERSION,recording_id:String(meta.recording_id),created_at:meta.created_at,
      participant_id:checked.participant_id,bottle_id:checked.bottle_id,tested_hand:checked.tested_hand,camera_facing:checked.camera_facing,
      source:meta.source||(meta.camera_facing==='file'?'file':'camera'),app_version:meta.app_version,protocol:{...C.PROTOCOL},completed:!!completed,
      measurement_scope:'video_feature_changes',force_estimation_enabled:false,analysis_location:'on_device_browser',
      rows,statistics:{},features:{}};
    if(typeof meta.source_recording_id==='string'&&meta.source_recording_id.length<=300)record.source_recording_id=meta.source_recording_id;
    for(const name of C.SIGNALS){
      const stat=describe(record,name);record.statistics[name]=stat;
      for(const key of ['p90','net_change','range','speed_abs_p90','speed_positive_p90','acceleration_abs_p90','release_residual','rest_noise','signal_to_noise','return_pct'])
        record.features[name+'_'+key]=stat[key];
    }
    const counts=Object.fromEntries(PHASES.map(p=>[p,rows.filter(r=>r.phase===p).length]));
    const issues=[],warnings=[];
    if(!completed)issues.push('interrupted_trial');
    if(counts.rest<4||counts.squeeze<8||counts.release<4)issues.push('too_few_frames_in_each_phase');
    if(rows.some((r,i)=>i&&r.t-rows[i-1].t>.75))warnings.push('analysis_gap_too_long');
    if(!rows.length||rows.at(-1).t<6.5)issues.push('release_recording_too_short');
    const channels=Object.fromEntries(Object.entries(CHANNELS).map(([key,name])=>[key,record.statistics[name]]));
    if(!channels.observed.supported)warnings.push('observed_bottle_width_not_supported');
    if(!channels.proxy.supported)warnings.push('proxy_not_supported');
    if(!channels.hand.supported)warnings.push('hand_not_supported');
    const observed=Object.values(channels).some(s=>s.supported);
    record.quality={issues,warnings,channels,frames:rows.length,phase_frames:counts,observation_usable:!issues.length&&observed,
      hand_valid_fraction:rows.length?rows.filter(r=>r.f.valid_hand===1).length/rows.length:0,
      axis_valid_fraction:rows.length?rows.filter(r=>r.f.valid_axis===1).length/rows.length:0,
      mean_processing_ms:mean(rows.map(r=>r.processing_ms))};
    return record;
  }
  function parseRecord(value){
    if(!value||!Array.isArray(value.rows)||value.rows.length>10000||!String(value.feature_version||'').startsWith('iphone_'))
      throw Error('PREX 아이폰 앱에서 내보낸 측정 JSON을 선택해 주세요.');
    const meta={...value,mode:'features'};
    if(!['environment','user','file'].includes(meta.camera_facing))throw Error('촬영 조건이 없는 파일입니다.');
    return buildRecord(value.rows,meta,value.completed??!value.quality?.issues?.includes('interrupted_trial'));
  }
  const cell=v=>'"'+String(typeof v==='string'&&/^[\s=+@-]/.test(v)?"'"+v:v??'').replace(/"/g,'""')+'"';
  function csv(record,summary=false){
    let lines;
    if(summary){
      lines=[['signal','phase','valid_frames','total_frames','coverage_pct','median','p10','p90','speed_abs_p90','acceleration_abs_p90']];
      for(const [name,s] of Object.entries(record.statistics))for(const p of PHASES){const v=s.phases[p];
        lines.push([name,p,v.valid,v.total,v.coverage_pct,v.median,v.p10,v.p90,s.speed_abs_p90,s.acceleration_abs_p90]);}
    }else{
      const names=[...C.SIGNALS,'valid_hand','valid_axis','valid_observed','valid_proxy'];
      lines=[['time_s','source_frame_id','trial_phase','participant_id','bottle_id','tested_hand','processing_ms',...names.map(n=>'feat_'+n)]];
      for(const r of record.rows)lines.push([r.t,r.id,r.phase,record.participant_id,record.bottle_id,record.tested_hand,r.processing_ms,...names.map(n=>r.f[n])]);
    }return '\uFEFF'+lines.map(r=>r.map(cell).join(',')).join('\r\n');
  }
  const api={VERSION,PHASES,LABELS,CHANNELS,signal,rates,describe,buildRecord,parseRecord,csv};
  root.PrexObservations=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
