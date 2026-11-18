/* Small local canvas plots; observations are never interpolated across missing data. */
(function(root){
  'use strict';
  const C=root.PrexCore,O=root.PrexObservations;
  const FINGERS={thumb:'엄지',index:'검지',middle:'중지',ring:'약지',pinky:'소지'};
  const COLORS=['#24485b','#2f759b','#78949f','#4c7e72','#816d91'];
  const fmt=(v,d=1)=>C.finite(v)?v.toFixed(d):'—';
  function jointLabel(name){const [finger,joint]=name.split('_');return FINGERS[finger]+' '+joint.toUpperCase();}
  function selection(view){
    if(view==='fingers')return Object.keys(FINGERS).map((name,i)=>({name:'hand_'+name+'_change_deg',label:FINGERS[name],color:COLORS[i]}));
    if(C.NAMES.includes(view))return [{name:'flexion_change_'+view+'_deg',label:jointLabel(view),color:COLORS[0]}];
    return [{name:'hand_flexion_change_mean_deg',label:'14개 관절 평균',color:COLORS[0]}];
  }
  function lines(record,kind,view,motion){
    const chosen=kind==='hand'?selection(view):[
      {name:'observed_width_reduction_pct',label:'직접 관측',color:'#2f759b'},
      {name:'proxy_width_reduction_pct',label:'관절 기반 추정',color:'#78949f',dashed:true}];
    return chosen.map(s=>{let values=O.signal(record,s.name);if(motion!=='change')values=O.rates(record.rows.map(r=>r.t),values)[motion];
      return {...s,values};});
  }
  function draw(canvas,record,kind,options={}){
    const width=options.width||canvas.clientWidth||320,height=options.height||230,scale=options.scale||Math.min(root.devicePixelRatio||1,2);
    canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);
    const ctx=canvas.getContext('2d');ctx.scale(scale,scale);ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);
    const motion=options.motion||'change',series=lines(record,kind,options.view||'mean',motion);
    const left=45,right=width-13,top=32,bottom=height-28,px=t=>left+(right-left)*t/7;
    ctx.font='10px -apple-system, sans-serif';ctx.textBaseline='middle';
    for(const [a,b,color,label] of [[0,2,'#f2f6f7','기준 파지'],[2,5,'#dee9ef','최대 노력 파지'],[5,7,'#e8f0f3','이완 및 회복']]){
      ctx.fillStyle=color;ctx.fillRect(px(a),top,px(b)-px(a),bottom-top);
      ctx.fillStyle='#526e7c';ctx.textAlign='center';ctx.fillText(label,px((a+b)/2),17);
    }
    const all=series.flatMap(s=>s.values).filter(C.finite);
    let low=Math.min(0,...all),high=Math.max(5,...all),span=high-low;low-=span*.08;high+=span*.12;
    const py=v=>bottom-(v-low)/(high-low)*(bottom-top);
    const suffix=motion==='speed'?'/s':motion==='acceleration'?'/s²':'';
    ctx.textAlign='left';ctx.fillStyle='#526e7c';ctx.fillText((kind==='hand'?'°':'%')+suffix,2,15);
    for(let i=0;i<4;i++){
      const v=low+(high-low)*i/3,y=py(v);ctx.strokeStyle='#d2dfe5';ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
      ctx.textAlign='right';ctx.fillStyle='#526e7c';ctx.fillText(fmt(v,Math.abs(v)>100?0:1),left-6,y);
    }
    ctx.strokeStyle='#9eb7c4';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(left,py(0));ctx.lineTo(right,py(0));ctx.stroke();
    ctx.textAlign='center';for(let t=0;t<=7;t++)ctx.fillText(t+'s',px(t),height-10);
    for(const seriesLine of series){
      ctx.strokeStyle=seriesLine.color;ctx.fillStyle=seriesLine.color;ctx.lineWidth=1.8;ctx.setLineDash(seriesLine.dashed?[5,4]:[]);ctx.beginPath();let previous=null;
      record.rows.forEach((row,i)=>{
        const v=seriesLine.values[i];if(!C.finite(v)){previous=null;return;}
        if(previous&&row.t-previous.t<=.35)ctx.lineTo(px(row.t),py(v));else ctx.moveTo(px(row.t),py(v));previous=row;
      });ctx.stroke();ctx.setLineDash([]);
      record.rows.forEach((row,i)=>{const v=seriesLine.values[i];if(C.finite(v)){ctx.beginPath();ctx.arc(px(row.t),py(v),1.8,0,Math.PI*2);ctx.fill();}});
    }
    if(!all.length){ctx.fillStyle='#607b8b';ctx.textAlign='center';ctx.font='12px -apple-system, sans-serif';ctx.fillText('유효한 관측값이 없습니다.',(left+right)/2,(top+bottom)/2);}
    return series;
  }
  function png(record,{view='mean',motion='change'}={}){
    const canvas=document.createElement('canvas');canvas.width=1600;canvas.height=1240;
    const x=canvas.getContext('2d');x.fillStyle='white';x.fillRect(0,0,1600,1240);x.fillStyle='#24485b';
    x.font='bold 36px -apple-system, sans-serif';x.fillText('PREX · 손 관절과 병 너비 변화',70,68);
    x.font='20px -apple-system, sans-serif';x.fillStyle='#526e7c';
    const condition=`${record.participant_id} · ${record.tested_hand==='left'?'왼손':'오른손'} · ${record.bottle_id}`;
    x.fillText(condition,70,110,1460);x.fillText(new Date(record.created_at).toLocaleString('ko-KR')+' · '+record.rows.length+'개 분석 프레임',70,145);
    for(const [kind,y,label] of [['hand',200,'손 관절'],['bottle',660,'페트병 너비']]){
      x.font='bold 25px -apple-system, sans-serif';x.fillStyle='#24485b';x.fillText(label,70,y);
      const chart=document.createElement('canvas'),series=draw(chart,record,kind,{view,motion,width:1460,height:360,scale:1});
      x.drawImage(chart,70,y+20);let lx=90;x.font='18px -apple-system, sans-serif';
      for(const line of series){x.fillStyle=line.color;x.fillRect(lx,y+405,22,3);x.fillText(line.label,lx+31,y+411);lx+=230;}
    }
    x.fillStyle='#607b8b';x.font='18px -apple-system, sans-serif';
    x.fillText('기준 파지 2초 → 최대 노력 파지 3초 → 이완 및 회복 2초. x축은 실제 분석 시각입니다.',70,1137);
    x.fillText('누락된 값은 선을 끊었습니다. 점선은 손 관절 기반 추정치입니다.',70,1170);
    x.fillText('영상 변화 관찰 자료 · 악력 또는 내부 압력 환산값 없음',70,1203);
    return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('그래프 이미지를 만들지 못했습니다.')),'image/png'));
  }
  const api={FINGERS,jointLabel,selection,draw,png};root.PrexCharts=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
