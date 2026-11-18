/* A classic worker is intentional: MediaPipe's WASM loader uses importScripts.
   Only small, downscaled frames cross this boundary, one at a time. */
importScripts('./assets.js','./core.js','./bottle.js');
const C=PrexCore,B=PrexBottle;
let handModel,bottleModel,canvas,ctx,ready=false,lastBottle=-Infinity,lastStamp=-1,lastInputStamp=null;
let tracker=new B.BottleTracker(),baseline=new B.BaselineCollector();
let shape=null;
async function initialize() {
  if(ready)return;
  if(typeof OffscreenCanvas==='undefined')throw Error('이 Safari는 별도 작업의 영상 분석을 지원하지 않습니다. iOS를 업데이트해 주세요.');
  postMessage({type:'loading',message:'아이폰에서 실행할 무료 AI 모델을 불러오는 중입니다.'});
  const {FilesetResolver,HandLandmarker,ObjectDetector}=await import(PREX_ASSETS.runtime);
  const files=await FilesetResolver.forVisionTasks(PREX_ASSETS.wasmBase);
  // CPU/WASM works without cross-origin isolation or a WebGPU requirement.
  handModel=await HandLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:PREX_ASSETS.hand,delegate:'CPU'},
    runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:.35,minHandPresenceConfidence:.35,minTrackingConfidence:.3});
  try {
    bottleModel=await ObjectDetector.createFromOptions(files,{baseOptions:{modelAssetPath:PREX_ASSETS.bottle,delegate:'CPU'},
      runningMode:'VIDEO',categoryAllowlist:['bottle'],scoreThreshold:.3,maxResults:3});
  } catch(error) {bottleModel=null;postMessage({type:'warning',message:'병 모델을 불러오지 못했습니다. 손 관절은 계속 분석합니다. '+String(error.message||error)});}
  ready=true;postMessage({type:'ready'});
}
function reset(){tracker.reset();baseline.reset();lastBottle=-Infinity;shape=null;lastInputStamp=null;}
function analyze(message) {
  const {bitmap,id,timestamp}=message,started=performance.now();
  try {
    const w=bitmap.width,h=bitmap.height;
    if(!canvas){canvas=new OffscreenCanvas(w,h);ctx=canvas.getContext('2d',{willReadFrequently:true});}
    if(shape!==`${w}x${h}`){canvas.width=w;canvas.height=h;reset();shape=`${w}x${h}`;}
    ctx.drawImage(bitmap,0,0,w,h);
    const step=lastInputStamp===null?1:Math.max(1,timestamp-lastInputStamp);
    const stamp=lastStamp<0?timestamp:lastStamp+step;lastStamp=stamp;lastInputStamp=timestamp;
    const result=handModel.detectForVideo(canvas,stamp);
    const landmarks=result.landmarks?.[0],angles=C.jointAngles(result.worldLandmarks?.[0]);
    const hand=landmarks&&angles?{points:landmarks.map(v=>[v.x*w,v.y*h]),angles,
      side:result.handedness?.[0]?.[0]?.categoryName||'unknown'}:null;
    let detections=null;
    if(bottleModel&&timestamp-lastBottle>=900) {
      const found=bottleModel.detectForVideo(canvas,stamp);lastBottle=timestamp;
      detections=found.detections.filter(d=>d.categories.some(c=>c.categoryName==='bottle')).map(d=>({
        x:d.boundingBox.originX,y:d.boundingBox.originY,width:d.boundingBox.width,height:d.boundingBox.height,
        score:Math.max(...d.categories.map(c=>c.score))}));
    }
    const image=ctx.getImageData(0,0,w,h),bottle=tracker.measure(image.data,w,h,hand,detections,timestamp);
    // Hand references remain usable when the independent bottle track is lost.
    baseline.update(hand,bottle,timestamp,tracker);
    const f=C.features(hand,bottle,baseline.reference);
    postMessage({type:'frame',id,timestamp,generation:message.generation,w,h,features:f,
      hand:hand?{points:hand.points,angles:hand.angles,side:hand.side}:null,
      bottle:{valid:bottle.valid,reason:bottle.reason,segments:bottle.segments,proxy:bottle.proxy,
        cap:bottle.pose?.cap,base:bottle.pose?.base},
      baselineReady:!!baseline.reference,baselineBusy:baseline.started!==null,baselineMessage:baseline.message,
      processingMs:performance.now()-started});
  } finally {bitmap.close();}
}
onmessage=async event=>{
  const m=event.data;
  try {
    if(m.type==='init')await initialize();
    else if(m.type==='reset')reset();
    else if(m.type==='baseline'){tracker.reference=null;baseline.request(m.timestamp,m.delay??3000);}
    else if(m.type==='frame') {
      if(!ready){m.bitmap.close();throw Error('AI 모델이 아직 준비되지 않았습니다.');}
      analyze(m);
    }
  } catch(error) {postMessage({type:'error',id:m.id,generation:m.generation,fatal:m.type==='init',message:String(error.message||error)});}
};
