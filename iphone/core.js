/* Landmarks and numerical camera features, shared by the UI, worker, and tests.
   Phone features have a separate version from the desktop recording pipeline. */
(function (root) {
  'use strict';
  const VERSION = 'iphone_observation_v2';
  const JOINTS = [[1,2,3],[2,3,4],[0,5,6],[5,6,7],[6,7,8],[0,9,10],[9,10,11],
    [10,11,12],[0,13,14],[13,14,15],[14,15,16],[0,17,18],[17,18,19],[18,19,20]];
  const NAMES = ['thumb_mcp','thumb_ip','index_mcp','index_pip','index_dip',
    'middle_mcp','middle_pip','middle_dip','ring_mcp','ring_pip','ring_dip','pinky_mcp','pinky_pip','pinky_dip'];
  const FINGERS = {thumb:[0,2],index:[2,5],middle:[5,8],ring:[8,11],pinky:[11,14]};
  const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],
    [10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
  const finite = n => typeof n === 'number' && Number.isFinite(n);
  const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
  function quantile(values, q = .5) {
    const v = values.filter(finite).sort((a,b)=>a-b);
    if (!v.length) return null;
    const p = (v.length-1)*q, i = Math.floor(p);
    return v[i] + (v[Math.min(i+1,v.length-1)]-v[i])*(p-i);
  }
  function mean(v) { v = v.filter(finite); return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null; }
  const distance = (a,b) => Math.hypot(a[0]-b[0],a[1]-b[1]);
  function jointAngles(world) {
    if (!world || world.length !== 21) return null;
    const p = world.map(v=>Array.isArray(v)?v:[v.x,v.y,v.z]);
    if (p.some(v=>v.length<3 || v.some(x=>!finite(x)))) return null;
    const out = JOINTS.map(([a,b,c])=>{
      const u=p[a].map((x,i)=>x-p[b][i]), v=p[c].map((x,i)=>x-p[b][i]);
      const d=Math.hypot(...u)*Math.hypot(...v);
      return d<1e-12 ? null : Math.acos(clamp(u.reduce((s,x,i)=>s+x*v[i],0)/d,-1,1))*180/Math.PI;
    });
    return out.every(finite) ? out : null;
  }
  function hull(points) {
    const p = points.map(v=>v.slice(0,2)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const side = a=>{const s=[];for(const v of a){while(s.length>=2&&cross(s.at(-2),s.at(-1),v)<=0)s.pop();s.push(v);}return s;};
    return [...side(p).slice(0,-1),...side(p.slice().reverse()).slice(0,-1)];
  }
  function geometry(points) {
    if (!points || points.length!==21) return {};
    const palm=distance(points[5],points[17]);
    if (palm<10) return {};
    const center=[0,1].map(i=>mean([0,5,9,13,17].map(j=>points[j][i]))), poly=hull(points);
    const area=Math.abs(poly.reduce((s,p,i)=>{const q=poly[(i+1)%poly.length];return s+p[0]*q[1]-p[1]*q[0];},0))/2;
    return {hand_hull_area_norm:area/palm**2,
      hand_tip_distance_norm:mean([4,8,12,16,20].map(i=>distance(points[i],center)))/palm,
      hand_thumb_index_norm:distance(points[4],points[8])/palm,
      hand_thumb_middle_norm:distance(points[4],points[12])/palm,
      hand_tip_span_norm:distance(points[8],points[20])/palm};
  }
  const SIGNALS = [...NAMES.map(n=>'angle_'+n+'_deg'),...NAMES.map(n=>'flexion_change_'+n+'_deg'),
    'hand_flexion_mean_deg','hand_flexion_change_mean_deg','hand_flexion_change_abs_mean_deg',
    'hand_flexion_change_positive_mean_deg',...Object.keys(FINGERS).map(n=>'hand_'+n+'_change_deg'),
    'hand_hull_area_norm','hand_tip_distance_norm','hand_thumb_index_norm','hand_thumb_middle_norm',
    'hand_tip_span_norm','hand_tip_distance_change_norm','observed_width_reduction_pct',
    'observed_area_reduction_pct','observed_dent_p95_pct','observed_left_dent_pct','observed_right_dent_pct',
    'observed_asymmetry_pct','observed_contour_rms_pct','observed_profile_iqr_pct',
    'proxy_width_reduction_pct','proxy_curvature_norm'];
  function features(hand, bottle, reference) {
    const out=Object.fromEntries(SIGNALS.map(n=>[n,null]));
    Object.assign(out,{valid_hand:0,valid_axis:0,valid_observed:0,valid_proxy:0});
    const angles=hand?.angles;
    if (angles?.length===14 && angles.every(a=>finite(a)&&a>=0&&a<=180)) {
      out.valid_hand=1;
      NAMES.forEach((n,i)=>out['angle_'+n+'_deg']=angles[i]);
      out.hand_flexion_mean_deg=mean(angles.map(a=>180-a));
      Object.assign(out,geometry(hand.points));
      if (reference?.angles && (!reference.side || hand.side===reference.side)) {
        const change=reference.angles.map((a,i)=>a-angles[i]);
        NAMES.forEach((n,i)=>out['flexion_change_'+n+'_deg']=change[i]);
        Object.assign(out,{hand_flexion_change_mean_deg:mean(change),
          hand_flexion_change_abs_mean_deg:mean(change.map(Math.abs)),
          hand_flexion_change_positive_mean_deg:mean(change.map(v=>Math.max(0,v)))});
        for(const [name,[a,b]] of Object.entries(FINGERS)) out['hand_'+name+'_change_deg']=mean(change.slice(a,b));
        const before=geometry(reference.points);
        if (finite(before.hand_tip_distance_norm)&&finite(out.hand_tip_distance_norm))
          out.hand_tip_distance_change_norm=before.hand_tip_distance_norm-out.hand_tip_distance_norm;
      }
    }
    if (bottle?.valid) {
      out.valid_axis=1;
      for(const n of SIGNALS.filter(n=>n.startsWith('observed_')||n.startsWith('proxy_')))
        if(finite(bottle.metrics?.[n]))out[n]=bottle.metrics[n];
      out.valid_observed=Number(finite(out.observed_width_reduction_pct));
      out.valid_proxy=Number(finite(out.proxy_width_reduction_pct));
    }
    return out;
  }
  function validateMetadata(meta) {
    for(const key of ['participant_id','bottle_id'])
      if(typeof meta[key]!=='string'||!meta[key].trim()||meta[key].length>100)throw Error('참가자와 병 조건을 입력해 주세요.');
    if(!['left','right'].includes(meta.tested_hand))throw Error('측정할 손을 선택해 주세요.');
    if(!['environment','user','file'].includes(meta.camera_facing))throw Error('촬영 조건이 올바르지 않습니다.');
    return {participant_id:meta.participant_id.trim(),bottle_id:meta.bottle_id.trim(),tested_hand:meta.tested_hand,camera_facing:meta.camera_facing,mode:'features'};
  }
  const PROTOCOL={rest:2,squeeze:3,release:2};
  function phase(t) {return t<0?'before':t<2?'rest':t<5?'squeeze':t<7?'release':'complete';}
  const api={VERSION,JOINTS,NAMES,FINGERS,CONNECTIONS,SIGNALS,PROTOCOL,finite,clamp,quantile,mean,distance,
    hull,geometry,jointAngles,features,validateMetadata,phase};
  root.PrexCore=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
