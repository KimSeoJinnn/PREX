import * as Store from './storage.js';
const C=globalThis.PrexCore,O=globalThis.PrexObservations,G=globalThis.PrexCharts,$=id=>document.getElementById(id);
const video=$('video'),overlay=$('overlay'),ctx=overlay.getContext('2d');
const capture=document.createElement('canvas'),captureCtx=capture.getContext('2d');
let worker,workerReady=false,initPromise,initResolve,initReject;
let stream,source='none',facing='environment',fileUrl,fileRunning=false,fileCancelled=false;
let generation=0,serial=0,pending=null,latest=null,active=false,nextAt=0,lastVideoTime=-1;
let trial=null,finishing=false,baselineConsumed=false,records=[],selected=null,offlineBusy=false;
let wakeLock=null,raf=0,starting=false,initTimer=null;
const format=(v,d=1)=>C.finite(v)?v.toFixed(d):'—';
function notice(text){$('notice').textContent=text;$('notice').hidden=!text;}
function metadata() {
  return C.validateMetadata({participant_id:$('participant').value.trim(),bottle_id:$('bottleId').value.trim(),
    tested_hand:$('hand').value,camera_facing:source==='file'?'file':facing,mode:'features'});
}
function lockFields(locked) {
  for(const id of ['participant','bottleId','hand','videoFile','importJson'])$(id).disabled=locked;
}
function buttons() {
  const locked=!!trial||finishing||fileRunning||starting;
  $('cameraStart').disabled=locked||active;
  $('cameraStop').disabled=(!active&&!starting)||finishing;
  $('flip').disabled=!active||locked;
  $('baseline').disabled=!active||!workerReady||locked||!latest;
  $('measure').disabled=!active||locked||!latest?.baselineReady||baselineConsumed||!latest?.hand;
  $('cancel').disabled=!trial&&!fileRunning;
  $('analyzeFile').disabled=fileRunning||!!trial||finishing||source!=='file';
  $('deleteRecord').disabled=locked||!selected;
  lockFields(locked);
}
function ensureWorker() {
  if(workerReady)return Promise.resolve();
  if(initPromise)return initPromise;
  initPromise=new Promise((resolve,reject)=>{initResolve=resolve;initReject=reject;});
  worker=new Worker('./inference-worker.js');
  initTimer=setTimeout(()=>handleWorkerError('모델 준비가 지연되었습니다. 인터넷 연결을 확인하고 카메라를 다시 켜 주세요.',true),120000);
  worker.onmessage=event=>{
    const m=event.data;
    if(m.type==='ready'){clearTimeout(initTimer);workerReady=true;initResolve();buttons();}
    if(m.type==='warning')notice(m.message);
    if(m.type==='loading')$('trackingHint').textContent=m.message;
    if(m.type==='frame') {
      const request=pending?.id===m.id?pending:null;
      if(request?.id===m.id)pending=null;
      if(m.generation===generation) {
        latest=m;nextAt=performance.now()+Math.max(0,m.processingMs*.12);
        if(trial&&m.timestamp>=trial.start&&m.timestamp<trial.start+7000)
          trial.rows.push({id:m.id,t:(m.timestamp-trial.start)/1000,phase:C.phase((m.timestamp-trial.start)/1000),f:m.features,processing_ms:m.processingMs});
        renderLive(m);buttons();
      }
      request?.resolve(m);
      if(finishing&&!pending)finishLive(true);
    }
    if(m.type==='error'&&(m.generation===undefined||m.generation===generation))handleWorkerError(m.message,m.fatal);
  };
  worker.onerror=event=>handleWorkerError(event.message||'영상 분석 작업이 종료되었습니다.',true);
  worker.postMessage({type:'init'});
  return initPromise;
}
function handleWorkerError(message,fatal=false) {
  const p=pending;pending=null;p?.reject(Error(message));notice(message);
  if(trial)finishLive(false);
  if(fatal) {
    clearTimeout(initTimer);
    initReject?.(Error(message));worker?.terminate();worker=null;workerReady=false;initPromise=null;
    stopCamera();
  }
  buttons();
}
async function sendFrame(timestamp) {
  if(pending)throw Error('이전 프레임을 분석 중입니다.');
  const w=video.videoWidth,h=video.videoHeight;
  if(!w||!h)throw Error('영상 프레임을 읽을 수 없습니다.');
  const scale=Math.min(512/w,512/h,1);
  capture.width=Math.max(1,Math.round(w*scale));capture.height=Math.max(1,Math.round(h*scale));
  captureCtx.drawImage(video,0,0,capture.width,capture.height);
  const id=++serial,gen=generation;
  const promise=new Promise((resolve,reject)=>{pending={id,resolve,reject,started:performance.now()};});
  try {
    const bitmap=await createImageBitmap(capture);
    if(gen!==generation){bitmap.close();throw Error('촬영 조건이 변경되었습니다.');}
    worker.postMessage({type:'frame',id,timestamp,generation:gen,bitmap},[bitmap]);
  } catch(error){if(pending?.id===id){const p=pending;pending=null;p.reject(error);}}
  return promise;
}
async function keepAwake() {
  try{wakeLock=await navigator.wakeLock?.request('screen');}catch{ /* optional */ }
}
function clearSource() {
  generation++;latest=null;baselineConsumed=false;
  if(pending){pending.reject(new DOMException('촬영 조건이 변경되었습니다.','AbortError'));pending=null;}
  stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;
  if(fileUrl){URL.revokeObjectURL(fileUrl);fileUrl=null;}
  video.removeAttribute('src');video.load();ctx.clearRect(0,0,overlay.width,overlay.height);
  worker?.postMessage({type:'reset'});lastVideoTime=-1;
  for(const id of ['liveHand','liveWidth','liveProxy'])$(id).textContent='—';
  for(const id of ['handBadge','bottleBadge','baselineBadge'])$(id).classList.remove('ok');
  resetLive();
}
function resetLive(){
  for(const id of ['liveHand','liveWidth','liveProxy'])$(id).textContent='—';
  for(const [id,text] of [['handBadge','손 대기'],['bottleBadge','병 대기'],['baselineBadge','기준 대기'],['fps','준비 중']]){$(id).textContent=text;$(id).classList.remove('ok');}
}
async function startCamera() {
  if(starting||trial||fileRunning||finishing)return;
  let token;
  notice('');
  try {
    if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)
      throw Error('Safari에서 HTTPS 주소로 열어야 카메라를 사용할 수 있습니다. 파일 앱의 HTML 또는 일반 HTTP 주소로는 촬영할 수 없습니다.');
    active=false;cancelAnimationFrame(raf);clearSource();token=generation;source='camera';starting=true;buttons();
    video.controls=false;video.muted=true;video.playsInline=true;
    // The native video keeps playing while the worker analyzes fewer frames.
    const obtained=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:facing},
      width:{ideal:720},height:{ideal:960},frameRate:{ideal:24,max:30}}});
    if(token!==generation){obtained.getTracks().forEach(t=>t.stop());return;}stream=obtained;
    video.srcObject=stream;await video.play();
    $('placeholder').hidden=true;$('fileControls').hidden=true;
    $('sourceBadge').textContent=facing==='environment'?'후면 카메라':'전면 카메라';
    $('cameraBox').style.aspectRatio=`${video.videoWidth} / ${video.videoHeight}`;
    stream.getVideoTracks()[0].addEventListener('ended',()=>{if(active)stopCamera();},{once:true});
    await ensureWorker();if(token!==generation)return;starting=false;active=true;nextAt=0;await keepAwake();buttons();pump();
  } catch(error){if(token===undefined||token===generation){notice(cameraError(error));stopCamera();}}
  finally{if(token===generation)starting=false;buttons();}
}
function cameraError(error) {
  if(error.name==='NotAllowedError')return '카메라 사용이 허용되지 않았습니다. Safari의 이 웹사이트 설정에서 카메라를 허용한 뒤 다시 켜 주세요.';
  if(error.name==='NotReadableError')return '다른 앱에서 카메라를 사용 중일 수 있습니다. 카메라를 쓰는 앱을 닫고 다시 시도해 주세요.';
  return error.message||String(error);
}
function stopCamera() {
  active=false;starting=false;cancelAnimationFrame(raf);
  if(trial)finishLive(false);
  stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;
  wakeLock?.release().catch(()=>{});wakeLock=null;
  if(source==='camera') {
    generation++;source='none';latest=null;worker?.postMessage({type:'reset'});
    $('placeholder').hidden=false;$('sourceBadge').textContent='카메라 꺼짐';
    ctx.clearRect(0,0,overlay.width,overlay.height);
    resetLive();
  }
  buttons();
}
function pump() {
  if(!active)return;
  const now=performance.now();
  if(!pending&&!finishing&&now>=nextAt&&video.readyState>=2&&video.currentTime!==lastVideoTime) {
    lastVideoTime=video.currentTime;nextAt=now+110;
    sendFrame(now).catch(error=>{if(active)notice(error.message);});
  }
  if(latest&&now-latest.timestamp>450)ctx.clearRect(0,0,overlay.width,overlay.height);
  raf=requestAnimationFrame(pump);
}
function renderLive(m) {
  overlay.width=m.w;overlay.height=m.h;ctx.clearRect(0,0,m.w,m.h);
  // File timestamps are media time; camera timestamps use performance.now().
  if(source==='camera'&&performance.now()-m.timestamp>450)return;
  function lines(segments,color,dashed=false) {
    ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.setLineDash(dashed?[4,4]:[]);ctx.beginPath();
    for(const [a,b] of segments){ctx.moveTo(...a);ctx.lineTo(...b);}ctx.stroke();ctx.setLineDash([]);
  }
  if(m.hand){lines(C.CONNECTIONS.map(([a,b])=>[m.hand.points[a],m.hand.points[b]]),'#8bdd8c');
    ctx.fillStyle='#bcf79d';for(const p of m.hand.points){ctx.beginPath();ctx.arc(...p,2,0,2*Math.PI);ctx.fill();}}
  lines(m.bottle.segments||[],'#4bb0f5');
  const proxy=m.bottle.proxy||[];
  for(let side=0;side<2;side++)lines(proxy.slice(1).map((p,i)=>[proxy[i][side],p[side]]),'#ed88bf',true);
  if(m.bottle.valid){ctx.fillStyle='#67b8ee';for(const p of [m.bottle.cap,m.bottle.base])if(p){ctx.beginPath();ctx.arc(...p,3.5,0,Math.PI*2);ctx.fill();}}
  for(const [id,ok,text] of [['handBadge',!!m.hand,m.hand?'손 관절 14개':'손 대기'],
    ['bottleBadge',m.bottle.valid,m.bottle.valid?'병 기준축 확인':'병 대기 · 손 기록 가능'],
    ['baselineBadge',m.baselineReady,m.baselineReady?'기준 준비됨':m.baselineBusy?'기준 수집 중':'기준 대기']]) {
    $(id).textContent=text;$(id).classList.toggle('ok',ok);
  }
  $('liveHand').textContent=format(m.features.hand_flexion_change_mean_deg);
  $('liveWidth').textContent=format(m.features.observed_width_reduction_pct);
  $('liveProxy').textContent=format(m.features.proxy_width_reduction_pct);
  $('fps').textContent=`분석 ${Math.min(9,1000/Math.max(110,m.processingMs)).toFixed(0)} fps · ${Math.round(m.processingMs)} ms`;
  $('trackingHint').textContent=m.bottle.valid?(m.hand?'같은 병과 손 자세를 유지하세요. 손에 가린 분홍선은 추정입니다.':'손 전체가 화면 안에 보이도록 해 주세요.'):
    (m.hand?'손 관절은 기록 중입니다. 병 너비를 보려면 뚜껑·바닥과 몸통 양옆을 보여 주세요.':m.bottle.reason);
  if(!trial&&!fileRunning&&!finishing)$('guideTitle').textContent=m.baselineMessage;
}
function newMeta() {
  const meta=metadata(),now=new Date();
  return {...meta,recording_id:'iphone_'+now.toISOString().replace(/[:.]/g,'-')+'_'+crypto.randomUUID().slice(0,8),
    created_at:now.toISOString(),protocol:C.PROTOCOL,app_version:PREX_ASSETS.version,source};
}
async function beginLive() {
  try {
    if(!active||pending&&performance.now()-pending.started>2000||!latest?.baselineReady||baselineConsumed||!latest.hand)
      throw Error('가볍게 쥔 상태에서 기준을 먼저 잡아 주세요.');
    if(performance.now()-latest.timestamp>1000)throw Error('영상 분석이 지연되고 있습니다. 카메라를 다시 켜 주세요.');
    const meta=newMeta();
    trial={meta,start:performance.now(),rows:[]};baselineConsumed=true;notice('');buttons();
  } catch(error){notice(error.message);}
}
function guideTick() {
  if(pending&&performance.now()-pending.started>20000)handleWorkerError('영상 분석이 멈췄습니다. 측정을 중단하고 카메라를 다시 켜 주세요.',true);
  if(!trial)return;
  const elapsed=(performance.now()-trial.start)/1000,p=C.phase(elapsed);
  const ends={rest:2,squeeze:5,release:7};
  $('guide').classList.toggle('squeeze',p==='squeeze');
  $('guideTitle').textContent={rest:'가볍게 잡고 힘을 빼세요.',squeeze:'최대한 세게 쥐세요.',release:'힘을 빼고 가볍게 잡으세요.',complete:'결과를 계산하고 저장하는 중입니다.'}[p];
  $('countdown').textContent=p==='complete'?'':Math.max(0,Math.ceil(ends[p]-elapsed));
  $('progress').style.width=C.clamp(elapsed/7*100,0,100)+'%';
  if(p==='complete'&&!finishing){finishing=true;buttons();if(!pending)finishLive(true);}
}
async function finishLive(completed) {
  if(!trial)return;
  const t=trial;trial=null;finishing=true;buttons();
  try{await finishRecord(t.meta,t.rows,completed);}
  catch(error){notice('결과 처리 실패: '+error.message);}
  finally{finishing=false;$('countdown').textContent='';$('guide').classList.remove('squeeze');
    $('guideTitle').textContent=completed?'측정 완료 · 다음 측정 전 기준을 다시 잡으세요.':'측정 중단 · 기준을 다시 잡고 재측정하세요.';buttons();}
}
async function finishRecord(meta,rows,completed) {
  const record=O.buildRecord(rows,meta,completed);
  try{record.saved=true;await Store.save(record,null);}
  catch(error){record.saved=false;notice('기기 저장소에 저장하지 못했습니다. 화면의 JSON 또는 CSV를 지금 내보내 주세요. '+error.message);}
  records=[record,...records.filter(r=>r.recording_id!==record.recording_id)];selected=record;renderHistory();renderRecord();
}
function waitEvent(target,event,timeout=12000) {
  return new Promise((resolve,reject)=>{
    const done=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(Error('아이폰에서 이 영상을 읽지 못했습니다. Safari에서 재생 가능한 영상을 선택해 주세요.'));};
    const timer=setTimeout(()=>{cleanup();reject(Error('영상을 읽는 시간이 너무 오래 걸립니다.'));},timeout);
    function cleanup(){clearTimeout(timer);target.removeEventListener(event,done);target.removeEventListener('error',bad);}
    target.addEventListener(event,done,{once:true});target.addEventListener('error',bad,{once:true});
  });
}
async function chooseFile(event) {
  const file=event.target.files?.[0];if(!file)return;
  try {
    if(trial||fileRunning)return;
    stopCamera();clearSource();source='file';video.autoplay=false;video.controls=true;video.muted=true;
    fileUrl=URL.createObjectURL(file);const loaded=waitEvent(video,'loadedmetadata');video.src=fileUrl;video.load();await loaded;
    if(!C.finite(video.duration))throw Error('영상 길이를 읽지 못했습니다. 짧은 일반 동영상을 선택해 주세요.');
    $('placeholder').hidden=true;$('sourceBadge').textContent='아이폰 저장 영상';
    $('cameraBox').style.aspectRatio=`${video.videoWidth} / ${video.videoHeight}`;
    $('fileName').textContent=file.name+' · '+format(video.duration)+'초 · 영상은 업로드하지 않습니다.';
    $('fileControls').hidden=false;buttons();
  } catch(error){notice(error.message);}
}
async function seek(t) {
  if(Math.abs(video.currentTime-t)<.002&&video.readyState>=2)return;
  const done=waitEvent(video,'seeked');video.currentTime=t;await done;
  if(Math.abs(video.currentTime-t)>.08)throw Error('영상의 지정한 시점으로 이동하지 못했습니다.');
}
async function analyzeFile() {
  let meta,rows=[],completed=false;
  try {
    if(source!=='file'||fileRunning)return;
    meta=newMeta();const squeezeAt=Number($('squeezeAt').value),start=squeezeAt-2,end=squeezeAt+5;
    if(!C.finite(squeezeAt)||squeezeAt<3||end>video.duration+.05||end>60)
      throw Error('쥐기 시작 시점은 3초 이상이어야 하며, 이후 5초간의 영상이 필요합니다. 첫 60초 안의 동작을 선택하세요.');
    fileRunning=true;fileCancelled=false;notice('');buttons();await ensureWorker();
    while(pending)await new Promise(r=>setTimeout(r,50));
    generation++;worker.postMessage({type:'reset'});video.pause();video.controls=false;
    await seek(0);await sendFrame(0);worker.postMessage({type:'baseline',timestamp:0,delay:0});
    for(let t=.125;t<end-.001;t+=.125) {
      if(fileCancelled)break;
      await seek(t);const m=await sendFrame(t*1000);
      if(t>=start&&!m.baselineReady)throw Error('손 기준을 잡지 못했습니다. 처음 3초 이상 힘을 빼고 손 전체가 보이는 영상을 사용하세요.');
      if(t>=start&&t<end)rows.push({id:m.id,t:t-start,phase:C.phase(t-start),f:m.features,processing_ms:m.processingMs});
      $('guideTitle').textContent=`아이폰에서 영상 분석 중 · ${Math.round(t/end*100)}%`;
      $('countdown').textContent='';$('progress').style.width=(t/end*100)+'%';
      // Yield for Safari UI/input/thermal scheduling between decoded frames.
      await new Promise(r=>setTimeout(r,0));
    }
    completed=!fileCancelled;
    if(rows.length)await finishRecord(meta,rows,completed);
    $('guideTitle').textContent=completed?'영상 분석 완료 · 아래에서 결과를 확인하세요.':'영상 분석을 중단했습니다.';
  } catch(error){notice(error.message);if(meta&&rows.length)await finishRecord(meta,rows,false);}
  finally{fileRunning=false;video.controls=true;buttons();}
}
const issues={interrupted_trial:'측정이 중단됨',too_few_frames_in_each_phase:'이완·쥐기·다시 이완 구간의 프레임 부족',
  bottle_axis_missing_too_often:'병 경계 추적이 자주 끊김',no_supported_squeeze_cycle:'기준 잡기 이후의 뚜렷한 쥐기 변화를 확인하지 못함',
  analysis_gap_too_long:'영상 분석 간격이 너무 길었음',release_recording_too_short:'마지막 이완 구간 부족',
  observed_bottle_width_not_supported:'관측 병 너비의 비교 자료 부족 · 손에 가린 부분은 실측에서 제외',
  no_visible_return_after_release:'힘을 푼 뒤 원래 형태로 돌아오는 변화가 뚜렷하지 않음',
  hand_not_supported:'손 관절의 구간별 유효값 부족',proxy_not_supported:'손 관절 기반 추정 너비의 구간별 유효값 부족'};
function renderHistory() {
  const select=$('history');select.replaceChildren();
  if(!records.length){select.add(new Option('저장된 측정이 없습니다.',''));return;}
  for(const r of records)select.add(new Option(`${new Date(r.created_at).toLocaleString('ko-KR')} · ${r.participant_id} · ${r.tested_hand==='left'?'왼손':'오른손'}`,r.recording_id));
  select.value=selected?.recording_id||records[0].recording_id;
}
function renderRecord() {
  const r=selected;$('resultContent').hidden=!r;$('emptyResult').hidden=!!r;if(!r)return;
  const q=r.quality;
  $('resultStatus').textContent=q.observation_usable?'관찰 결과 준비됨':r.completed?'부분 관측 · 누락 확인':'중단된 관찰';
  $('resultDetail').textContent='손 관절·직접 관측한 병 너비·관절 기반 추정치를 각각 확인하세요. 병 너비가 누락되어도 손 관절 값은 유지합니다.';
  for(const [id,key] of [['resultHand','hand'],['resultWidth','observed'],['resultProxy','proxy']])$(id).textContent=format(q.channels[key].p90);
  $('quality').classList.toggle('bad',!q.observation_usable);
  $('quality').textContent=`${r.saved===false?'기기에 저장되지 않음 · 지금 내보내세요':'이 기기에 저장됨'} · 분석 ${q.frames}프레임 · 손 ${Math.round(q.hand_valid_fraction*100)}% · 병 ${Math.round(q.axis_valid_fraction*100)}%\n`+
    [...q.issues,...q.warnings].map(k=>issues[k]||k).join('\n');
  $('coverageRows').replaceChildren();
  for(const [key,label] of [['hand','손 관절'],['observed','관측 너비'],['proxy','추정 너비']]){
    const stat=q.channels[key];appendRow('coverageRows',[label,...O.PHASES.map(p=>`${stat.phases[p].valid}/${stat.phases[p].total} (${format(stat.phases[p].coverage_pct,0)}%)`)]);
  }
  $('featureRows').replaceChildren();
  for(const name of C.SIGNALS){const s=r.statistics[name];appendRow('featureRows',[featureLabel(name),format(s.p90),format(s.speed_abs_p90),format(s.release_residual)]);}
  $('featureCount').textContent=`${C.SIGNALS.length}개 신호 · ${Object.keys(r.features).length}개 요약값. 요약 CSV에는 세 구간의 관측 수·변화량·속도를 내보냅니다.`;
  $('frameSlider').max=Math.max(0,r.rows.length-1);$('frameSlider').value=0;renderFrame();
  drawChart(r);
  buttons();
}
function drawChart(record) {
  const mode=$('motion').value,options={view:$('handView').value,motion:mode==='speed'?'speed':mode==='acceleration'?'acceleration':'change'};
  const series=G.draw($('handChart'),record,'hand',options);G.draw($('bottleChart'),record,'bottle',options);
  $('handLegend').replaceChildren();for(const s of series){const span=document.createElement('span'),dot=document.createElement('i');
    dot.style.background=s.color;span.append(dot,document.createTextNode(s.label));$('handLegend').append(span);}
}
function appendRow(id,values){const tr=document.createElement('tr');for(const value of values)tr.append(Object.assign(document.createElement('td'),{textContent:value}));$(id).append(tr);}
function featureLabel(name){
  const joint=C.NAMES.find(n=>name.includes(n));if(joint)return G.jointLabel(joint)+(name.startsWith('angle_')?' 각도 변화':' 굽힘 변화');
  const labels={hand_flexion_mean_deg:'전체 굽힘',hand_flexion_change_mean_deg:'전체 굽힘 변화',hand_flexion_change_abs_mean_deg:'전체 변화 절댓값',hand_flexion_change_positive_mean_deg:'굽힌 방향 변화',
    observed_width_reduction_pct:'관측 너비 감소 (%)',observed_area_reduction_pct:'가시 면적 감소 (%)',proxy_width_reduction_pct:'추정 너비 감소 (%)',
    hand_hull_area_norm:'손 윤곽 면적 (정규화)',hand_tip_distance_norm:'손끝–손바닥 거리 (정규화)',hand_thumb_index_norm:'엄지–검지 거리 (정규화)',hand_thumb_middle_norm:'엄지–중지 거리 (정규화)',hand_tip_span_norm:'손끝 간격 (정규화)',hand_tip_distance_change_norm:'손끝 거리 변화 (정규화)',
    observed_dent_p95_pct:'국소 함몰 (%)',observed_left_dent_pct:'왼쪽 함몰 (%)',observed_right_dent_pct:'오른쪽 함몰 (%)',observed_asymmetry_pct:'좌우 비대칭 (%)',observed_contour_rms_pct:'윤곽 변위 RMS (%)',observed_profile_iqr_pct:'높이별 변형 차이 (%)',proxy_curvature_norm:'추정 너비 곡률'};
  for(const [finger,label] of Object.entries(G.FINGERS))if(name===`hand_${finger}_change_deg`)return label+' 평균 굽힘';
  return labels[name]||name;
}
function renderFrame(){
  if(!selected)return;const i=Number($('frameSlider').value),r=selected.rows[i];$('jointRows').replaceChildren();
  $('frameLabel').textContent=r?`${i+1}/${selected.rows.length} · ${format(r.t,2)}초 · ${O.LABELS[r.phase]}`:'유효 프레임 없음';
  for(const name of C.NAMES){const key='flexion_change_'+name+'_deg',s=selected.statistics[key];
    appendRow('jointRows',[G.jointLabel(name),format(r?.f['angle_'+name+'_deg'])+'°',format(O.signal(selected,key)[i])+'°',format(s.p90)+'°',format(s.speed_abs_p90)+'°/s']);}
}
async function exportData(kind) {
  if(!selected)return;
  const file=kind==='png'?new File([await G.png(selected,{view:$('handView').value,motion:$('motion').value})],selected.recording_id+'.png',{type:'image/png'}):
    new File([kind==='json'?JSON.stringify(selected,null,2):O.csv(selected,kind==='features')],
    selected.recording_id+(kind==='features'?'_summary.csv':'.'+kind),{type:kind==='json'?'application/json':'text/csv;charset=utf-8'});
  try {
    if(navigator.canShare?.({files:[file]})){try{await navigator.share({files:[file],title:'PREX 관찰 결과'});return;}catch(error){if(error.name==='AbortError')return;}}
    const url=URL.createObjectURL(file),a=document.createElement('a');a.href=url;a.download=file.name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
  } catch(error){if(error.name!=='AbortError')notice('내보내기 실패: '+error.message);}
}
async function offlineRequest(prepare=false) {
  if(!navigator.serviceWorker||!window.isSecureContext)throw Error('HTTPS 주소에서 Safari로 열어 주세요.');
  const registration=await Promise.race([navigator.serviceWorker.ready,new Promise((_,reject)=>setTimeout(()=>reject(Error('오프라인 준비 작업을 시작하지 못했습니다. 앱을 다시 열어 주세요.')),15000))]);
  return new Promise((resolve,reject)=>{
    const channel=new MessageChannel(),timer=setTimeout(()=>{channel.port1.close();reject(Error('다운로드가 지연되었습니다. 인터넷 연결을 확인하고 다시 눌러 주세요.'));},240000);
    channel.port1.onmessage=e=>{
      const m=e.data;
      if(m.type==='progress')$('offlineStatus').textContent=`앱과 AI 모델을 기기에 보관하는 중 · ${m.completed}/${m.total}`;
      else if(m.type==='done'){clearTimeout(timer);channel.port1.close();resolve(m);}
      else if(m.type==='error'){clearTimeout(timer);channel.port1.close();reject(Error(m.message));}
    };
    registration.active.postMessage({type:prepare?'prepare-offline':'offline-status'},[channel.port2]);
  });
}
$('cameraStart').onclick=startCamera;$('cameraStop').onclick=stopCamera;
$('flip').onclick=()=>{if(trial||fileRunning)return;facing=facing==='environment'?'user':'environment';startCamera();};
$('baseline').onclick=()=>{if(!active||trial)return;baselineConsumed=false;latest.baselineReady=false;
  worker.postMessage({type:'baseline',timestamp:performance.now(),delay:3000});notice('');buttons();};
$('measure').onclick=beginLive;
$('cancel').onclick=()=>{if(fileRunning)fileCancelled=true;else finishLive(false);};
$('videoFile').onchange=chooseFile;$('analyzeFile').onclick=analyzeFile;
$('history').onchange=()=>{selected=records.find(r=>r.recording_id===$('history').value);renderRecord();};
$('exportJson').onclick=()=>exportData('json').catch(e=>notice(e.message));$('exportCsv').onclick=()=>exportData('csv').catch(e=>notice(e.message));
$('exportFeatures').onclick=()=>exportData('features').catch(e=>notice(e.message));$('exportPng').onclick=()=>exportData('png').catch(e=>notice(e.message));
$('frameSlider').oninput=renderFrame;for(const id of ['handView','motion'])$(id).onchange=()=>{if(selected)drawChart(selected);};
$('importJson').onchange=async event=>{
  const file=event.target.files?.[0];if(!file)return;
  try{if(file.size>10*1024*1024)throw Error('10MB 이하의 PREX 측정 JSON을 선택해 주세요.');
    const record=O.parseRecord(JSON.parse(await file.text()));
    const sourceId=record.recording_id;record.recording_id='iphone_import_'+crypto.randomUUID();record.source_recording_id=sourceId;
    record.saved=true;await Store.save(record,null);records=[record,...records];selected=record;renderHistory();renderRecord();notice('관찰 결과를 이 기기에 불러왔습니다.');
  }catch(error){notice('불러오기 실패: '+error.message);}finally{event.target.value='';}
};
$('deleteRecord').onclick=async()=>{
  if(!selected||trial||fileRunning||finishing||!confirm('선택한 관찰 결과를 이 기기에서 삭제할까요? 내보낸 파일은 유지됩니다.'))return;
  try{await Store.remove(selected.recording_id);records=records.filter(r=>r.recording_id!==selected.recording_id);selected=records[0]||null;renderHistory();renderRecord();buttons();}
  catch(error){notice('삭제 실패: '+error.message);}
};
for(const name of C.NAMES)$('jointOptions').append(new Option(G.jointLabel(name),name));
for(const id of ['participant','bottleId','hand'])$(id).addEventListener('change',()=>{
  if(active&&['participant','bottleId','hand'].includes(id)){baselineConsumed=true;worker?.postMessage({type:'reset'});}
  buttons();
});
$('offline').onclick=async()=>{
  if(offlineBusy)return;offlineBusy=true;$('offline').disabled=true;
  try{const m=await offlineRequest(true);$('offlineStatus').textContent=m.ready?'오프라인 준비 완료 · 이제 인터넷 없이 기기에서 분석할 수 있습니다.':'일부 파일이 아직 없습니다. 다시 준비해 주세요.';}
  catch(error){$('offlineStatus').textContent=error.message;}
  finally{offlineBusy=false;$('offline').disabled=false;}
};
document.addEventListener('visibilitychange',()=>{
  if(document.hidden){fileCancelled=true;if(active||starting)stopCamera();}
});
window.addEventListener('pagehide',()=>{fileCancelled=true;stopCamera();});
window.addEventListener('resize',()=>{if(selected)drawChart(selected);});
setInterval(guideTick,70);
try {
  for(const record of await Store.trials()){try{const parsed=O.parseRecord(record);parsed.saved=true;records.push(parsed);}catch{/* Preserve unreadable legacy records in storage. */}}
  selected=records[0]||null;renderHistory();renderRecord();
} catch(error){notice('브라우저 저장소를 열지 못했습니다. 측정 결과를 JSON·CSV로 내보내세요. '+error.message);}
buttons();
if('serviceWorker' in navigator&&window.isSecureContext) {
  navigator.serviceWorker.register('./sw.js',{updateViaCache:'none'}).then(()=>offlineRequest()).then(m=>{
    if(m.ready)$('offlineStatus').textContent='오프라인 준비 완료 · 모델 파일이 이 기기에 보관되어 있습니다.';
  }).catch(error=>{$('offlineStatus').textContent='오프라인 준비를 확인하지 못했습니다. '+error.message;});
}
