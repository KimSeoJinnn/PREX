/* IndexedDB transactions keep a calibration and its source trial together. */
let promise;
function database() {
  if(!promise)promise=new Promise((resolve,reject)=>{
    const request=indexedDB.open('prex-iphone-v1',1);
    request.onupgradeneeded=()=>{
      const db=request.result;db.createObjectStore('trials',{keyPath:'recording_id'});
      db.createObjectStore('calibrations',{keyPath:'key'});
    };
    request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);
    request.onblocked=()=>reject(Error('다른 PREX 탭을 닫은 다음 다시 열어 주세요.'));
  });
  return promise;
}
export async function calibration(key) {
  const db=await database();
  return new Promise((resolve,reject)=>{
    const request=db.transaction('calibrations').objectStore('calibrations').get(key);
    request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>reject(request.error);
  });
}
export async function save(record,artifact) {
  const db=await database();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(artifact?['trials','calibrations']:['trials'],'readwrite');
    tx.objectStore('trials').put(record);
    if(artifact)tx.objectStore('calibrations').put(artifact);
    tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||Error('저장 취소'));
  });
}
export async function trials() {
  const db=await database();
  return new Promise((resolve,reject)=>{
    const request=db.transaction('trials').objectStore('trials').getAll();
    request.onsuccess=()=>resolve(request.result.sort((a,b)=>b.created_at.localeCompare(a.created_at)));
    request.onerror=()=>reject(request.error);
  });
}
export async function remove(recordingId) {
  const db=await database();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction('trials','readwrite');tx.objectStore('trials').delete(recordingId);
    tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||Error('삭제 취소'));
  });
}
