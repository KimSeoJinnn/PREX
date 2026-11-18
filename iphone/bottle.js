/* Visible-edge measurements, with explicit hand exclusion. AI boxes are only
   search regions. Hidden widths are a separate joint-derived proxy. */
(function (root) {
  'use strict';
  const C=root.PrexCore;
  const {finite,clamp,quantile,mean,distance,hull}=C;
  const count=31;
  function segmentDistance(p,a,b) {
    const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy;
    const t=d?clamp(((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d,0,1):0;
    return distance(p,[a[0]+t*dx,a[1]+t*dy]);
  }
  function inside(p,poly) {
    let ok=false;
    for(let i=0,j=poly.length-1;i<poly.length;j=i++) {
      const a=poly[i],b=poly[j];
      if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])ok=!ok;
    }
    return ok;
  }
  function handMask(points) {
    if(!points?.length)return ()=>false;
    const poly=hull(points),radius=Math.max(5,distance(points[5],points[17])*.14);
    return p=>inside(p,poly)||poly.some((v,i)=>segmentDistance(p,v,poly[(i+1)%poly.length])<radius)
      ||C.CONNECTIONS.some(([a,b])=>segmentDistance(p,points[a],points[b])<radius);
  }
  function pose(cap,base) {
    const length=distance(cap,base),v=[(base[0]-cap[0])/length,(base[1]-cap[1])/length],u=[v[1],-v[0]];
    return {cap,base,length,v,u,
      point:(x,y)=>[cap[0]+length*(u[0]*x+v[0]*y),cap[1]+length*(u[1]*x+v[1]*y)],
      local:p=>[((p[0]-cap[0])*u[0]+(p[1]-cap[1])*u[1])/length,((p[0]-cap[0])*v[0]+(p[1]-cap[1])*v[1])/length]};
  }
  function hullSlice(poly,y) {
    const hits=[];
    for(let i=0;i<poly.length;i++) {
      const a=poly[i],b=poly[(i+1)%poly.length];
      if((a[1]<=y&&b[1]>y)||(b[1]<=y&&a[1]>y))hits.push(a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1]));
    }
    return hits.length>=2?[Math.min(...hits),Math.max(...hits)]:null;
  }
  function grayImage(rgba,w,h) {
    const g=new Float32Array(w*h);
    for(let i=0,j=0;i<g.length;i++,j+=4)g[i]=rgba[j]*.299+rgba[j+1]*.587+rgba[j+2]*.114;
    return g;
  }
  function pixel(g,w,h,p) {
    const x=Math.round(p[0]),y=Math.round(p[1]);
    return x<1||y<1||x>=w-1||y>=h-1?null:g[y*w+x];
  }
  function edge(g,w,h,p,u) {
    const a=pixel(g,w,h,[p[0]+u[0]*2,p[1]+u[1]*2]);
    const b=pixel(g,w,h,[p[0]-u[0]*2,p[1]-u[1]*2]);
    return a===null||b===null?0:Math.abs(a-b);
  }
  function endpoint(g,w,h,box,isBase,blocked) {
    const {x,y,width:bw,height:bh}=box;
    const lo=y+bh*(isBase?.80:.01),hi=y+bh*(isBase?.99:.18);
    let best=null;
    for(let yy=Math.max(2,Math.round(lo));yy<Math.min(h-2,hi);yy+=2) {
      let score=0,weight=0,sumX=0,n=0;
      for(let x0=Math.max(2,Math.round(x+bw*.15));x0<Math.min(w-2,x+bw*.85);x0+=2) {
        if(blocked([x0,yy]))continue;
        const e=edge(g,w,h,[x0,yy],[0,1]);score+=e;n++;
        if(e>12){weight+=e;sumX+=x0*e;}
      }
      if(n<bw*.12||weight<40)continue;
      score/=n;
      if(!best||score>best.score)best={score,point:[sumX/weight,yy]};
    }
    return best&&best.score>=7?best.point:null;
  }
  class BottleTracker {
    constructor(){this.reset();}
    reset(){this.previous=null;this.reference=null;this.box=null;this.boxAt=-Infinity;this.stable=0;}
    setReference(result,hand) {
      const p=result.pose,canonical=hand.points.map(p.local),poly=hull(canonical);
      const v=canonical.map(x=>x[1]),low=Math.max(.25,Math.min(...v)),high=Math.min(.9,Math.max(...v));
      const upper=result.profiles.filter(r=>r.y<low-.015&&finite(r.left)&&finite(r.right));
      const lower=result.profiles.filter(r=>r.y>high+.015&&finite(r.left)&&finite(r.right));
      let proxy=null;
      if(upper.length>=2&&lower.length>=2) {
        const a=upper.at(-1),b=lower[0];
        const rows=[];
        for(let i=1;i<16;i++) {
          const y=low+(high-low)*i/16,slice=hullSlice(poly,y);
          if(!slice)continue;
          const t=(y-a.y)/(b.y-a.y),left=a.left+(b.left-a.left)*t,right=a.right+(b.right-a.right)*t;
          if(left>=right)continue;
          rows.push({y,left,right,gapLeft:slice[0]-left,gapRight:right-slice[1]});
        }
        if(rows.length>=5)proxy=rows;
      }
      this.reference={profiles:result.profiles.map(r=>({...r})),proxy,palm:distance(hand.points[5],hand.points[17])/p.length,
        side:hand.side,axisLength:p.length,center:p.point(0,.5)};
    }
    measure(rgba,w,h,hand,detections,now) {
      const blocked=handMask(hand?.points);
      if(detections!==null) {
        const candidates=detections.filter(b=>b.score>=.3&&b.height>=h*.25&&b.width>=w*.06&&b.height>b.width*1.25);
        if(candidates.length!==1) {this.box=null;this.stable=0;}
        else {this.box=candidates[0];this.boxAt=now;}
      }
      const box=this.box;
      const empty=reason=>({valid:false,reason,profiles:[],segments:[],proxy:[],metrics:{},box});
      if(!box||now-this.boxAt>1700)return empty('뚜껑과 바닥이 보이도록 병 전체를 보여 주세요.');
      const g=grayImage(rgba,w,h),cap=endpoint(g,w,h,box,false,blocked),base=endpoint(g,w,h,box,true,blocked);
      if(!cap||!base||base[1]-cap[1]<box.height*.6)return empty('뚜껑 또는 바닥의 경계가 불분명합니다.');
      let p=pose(cap,base);
      if(p.length<80||Math.abs(p.v[0])>.55)return empty('병을 세우고 카메라를 고정해 주세요.');
      const moved=this.previous&&(distance(p.cap,this.previous.cap)>p.length*.12||distance(p.base,this.previous.base)>p.length*.12);
      if(moved){this.stable=0;this.reference=null;}
      const profiles=[],segments=[];
      const expected=clamp(box.width/p.length/2,.055,.3);
      for(let i=0;i<count;i++) {
        const y=.22+i*.70/(count-1),row={y,left:null,right:null};
        for(const sign of [-1,1]) {
          const ref=this.reference?.profiles[i],expect=ref&&finite(sign<0?ref.left:ref.right)?Math.abs(sign<0?ref.left:ref.right):expected;
          let best=null;
          const start=Math.max(.025,expect*.5),stop=Math.min(.42,expect*1.5);
          for(let x=start;x<stop;x+=1.5/p.length) {
            const point=p.point(sign*x,y);
            if(blocked(point))continue;
            const strength=edge(g,w,h,point,p.u);
            if(strength<14)continue;
            const score=strength-Math.abs(x-expect)*95;
            if(!best||score>best.score)best={x:sign*x,score};
          }
          if(best)row[sign<0?'left':'right']=best.x;
        }
        if(finite(row.left)&&finite(row.right)) {
          const width=row.right-row.left;
          if(width<.07||width>expected*3||Math.abs(row.right+row.left)>.12)row.left=row.right=null;
        }
        profiles.push(row);
      }
      // Isolated high-contrast background/skin pixels cannot become an outline.
      for(const key of ['left','right']) {
        const values=profiles.map(r=>r[key]);
        for(let i=0;i<count;i++)if(finite(values[i])) {
          const neighbors=values.slice(Math.max(0,i-2),Math.min(count,i+3)).filter(finite);
          if(neighbors.length<2||Math.abs(values[i]-quantile(neighbors))>.035)profiles[i][key]=null;
        }
      }
      const pairs=profiles.filter(r=>finite(r.left)&&finite(r.right));
      const upper=pairs.filter(r=>r.y<.45),lower=pairs.filter(r=>r.y>.70);
      // The cap/base axis can remain valid while transparent body edges are
      // hidden. Width metrics below still require actual paired observations.
      this.stable=moved?1:this.stable+1;this.previous=p;
      for(const key of ['left','right'])for(let i=1;i<count;i++) {
        const a=profiles[i-1],b=profiles[i];
        if(finite(a[key])&&finite(b[key])&&Math.abs(a[key]-b[key])<.05)
          segments.push([p.point(a[key],a.y),p.point(b[key],b.y)]);
      }
      const metrics={},proxy=[],ref=this.reference;
      if(ref) {
        const reductions=[],lefts=[],rights=[],before=[],after=[];
        for(let i=0;i<count;i++) {
          const a=ref.profiles[i],b=profiles[i];
          if(![a.left,a.right,b.left,b.right].every(finite))continue;
          const width=a.right-a.left;
          if(width<.05)continue;
          reductions.push(100*(1-(b.right-b.left)/width));
          lefts.push(100*(b.left-a.left)/width);rights.push(100*(a.right-b.right)/width);
          before.push(width);after.push(b.right-b.left);
        }
        if(reductions.length>=6) Object.assign(metrics,{
          observed_width_reduction_pct:quantile(reductions),
          observed_area_reduction_pct:100*(1-after.reduce((a,b)=>a+b,0)/before.reduce((a,b)=>a+b,0)),
          observed_dent_p95_pct:quantile(reductions,.95),observed_left_dent_pct:quantile(lefts,.95),
          observed_right_dent_pct:quantile(rights,.95),
          observed_asymmetry_pct:Math.abs(quantile(lefts,.95)-quantile(rights,.95)),
          observed_contour_rms_pct:Math.sqrt(mean(reductions.map(x=>x*x))),
          observed_profile_iqr_pct:quantile(reductions,.75)-quantile(reductions,.25)});
        const palm=hand?distance(hand.points[5],hand.points[17])/p.length:0;
        if(hand&&hand.side===ref.side&&ref.proxy&&palm/ref.palm>.8&&palm/ref.palm<1.2) {
          const poly=hull(hand.points.map(p.local)),changes=[],widths=[];
          for(const r of ref.proxy) {
            const slice=hullSlice(poly,r.y);
            if(!slice)continue;
            const left=slice[0]-r.gapLeft,right=slice[1]+r.gapRight,width=right-left;
            if(width<=.025||width> (r.right-r.left)*1.5)continue;
            proxy.push([p.point(left,r.y),p.point(right,r.y)]);widths.push(width);
            changes.push(100*(1-width/(r.right-r.left)));
          }
          if(changes.length>=5) {
            metrics.proxy_width_reduction_pct=quantile(changes);
            metrics.proxy_curvature_norm=widths.length>=3?quantile(widths.slice(2).map((v,i)=>Math.abs(v-2*widths[i+1]+widths[i])))/quantile(widths):null;
          } else proxy.length=0;
        }
      }
      return {valid:true,reason:pairs.length<6?'병 축 추적 중 · 몸통 너비 관측 부족':this.stable>=3?'병 경계 추적 중':'병을 잠시 고정해 주세요.',
        stable:this.stable,pose:p,profiles,segments,proxy,metrics,box};
    }
  }
  class BaselineCollector {
    constructor(){this.reset();}
    reset(){this.started=null;this.samples=[];this.reference=null;this.message='가볍게 쥔 상태에서 기준 잡기를 눌러 주세요.';}
    request(now,delay=3000){this.reset();this.started=now+delay;}
    update(hand,bottle,now,tracker) {
      if(this.started===null)return;
      if(now<this.started){this.message='힘을 빼고 같은 자세를 유지하세요 · '+Math.ceil((this.started-now)/1000)+'초';return;}
      if(!hand?.angles) {
        this.samples=[];this.message='손 전체가 보이도록 유지해 주세요.';return;
      }
      const last=this.samples.at(-1);
      const unstable=last&&(now-last.now>500||hand.side!==last.hand.side||
        Math.max(...hand.angles.map((a,i)=>Math.abs(a-last.hand.angles[i])))>12||
        (bottle.valid&&last.bottle.valid&&distance(bottle.pose.point(0,.5),last.bottle.pose.point(0,.5))>bottle.pose.length*.025));
      if(unstable)this.samples=[];
      this.samples.push({hand,bottle,now});
      this.samples=this.samples.filter(s=>now-s.now<=1600);
      this.message='기준 수집 중 · 힘을 빼고 그대로 유지하세요.';
      if(this.samples.length<6||now-this.samples[0].now<700)return;
      const selected=this.samples.at(-1);
      const angles=C.NAMES.map((_,i)=>quantile(this.samples.map(s=>s.hand.angles[i])));
      const points=Array.from({length:21},(_,i)=>[0,1].map(j=>quantile(this.samples.map(s=>s.hand.points[i][j]))));
      this.reference={angles,points,side:hand.side};
      const visible=selected.bottle.valid&&selected.bottle.profiles.filter(r=>finite(r.left)&&finite(r.right)).length>=6;
      if(visible)tracker.setReference(selected.bottle,{...hand,points});
      this.started=null;this.message=visible?'손·병 기준 준비 완료 · 관찰을 시작하세요.':'손 기준 준비 완료 · 병 너비 기준은 확보되지 않았습니다.';
    }
  }
  const api={BottleTracker,BaselineCollector,pose,handMask,hullSlice,grayImage};
  root.PrexBottle=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
