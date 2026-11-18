/* All analysis is local. This worker only GETs static application/model files. */
importScripts('./assets.js');
const SHELL='prex-shell-'+PREX_ASSETS.version,MODELS='prex-models-0.10.21-v1';
const shell=['./','./index.html','./style.css','./app.js','./core.js','./observations.js','./charts.js','./bottle.js','./assets.js',
  './inference-worker.js','./storage.js','./manifest.webmanifest','./icon.svg','./icon-180.png','./icon-192.png','./icon-512.png'];
const allowed=new Set(PREX_ASSETS.files);
self.addEventListener('install',event=>event.waitUntil((async()=>{
  const cache=await caches.open(SHELL);await cache.addAll(shell);await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  for(const key of await caches.keys())if(key.startsWith('prex-shell-')&&key!==SHELL)await caches.delete(key);
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  const r=event.request,url=new URL(r.url);
  if(r.method!=='GET'||(!allowed.has(r.url)&&url.origin!==self.location.origin))return;
  event.respondWith((async()=>{
    const cache=await caches.open(allowed.has(r.url)?MODELS:SHELL),found=await cache.match(r.url);
    if(found)return found;
    const response=await fetch(r);
    if(response.ok&&(allowed.has(r.url)||shell.some(p=>new URL(p,self.location).href===r.url)))
      await cache.put(r.url,response.clone());
    return response;
  })());
});
self.addEventListener('message',event=>{
  if(!['prepare-offline','offline-status'].includes(event.data?.type))return;
  event.waitUntil((async()=>{
    const reply=data=>event.ports[0]?.postMessage(data),cache=await caches.open(MODELS);
    try {
      let completed=0;
      for(const url of PREX_ASSETS.files) {
        if(!(await cache.match(url))&&event.data.type==='prepare-offline') {
          const response=await fetch(url,{mode:'cors',credentials:'omit'});
          if(!response.ok)throw Error('모델 파일 다운로드 실패 ('+response.status+')');
          await cache.put(url,response);
        }
        if(await cache.match(url))completed++;
        if(event.data.type==='prepare-offline')reply({type:'progress',completed,total:PREX_ASSETS.files.length});
      }
      reply({type:'done',ready:completed===PREX_ASSETS.files.length,completed,total:PREX_ASSETS.files.length});
    } catch(error){reply({type:'error',message:String(error.message||error)});}
  })());
});
