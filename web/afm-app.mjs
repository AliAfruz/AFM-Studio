const $=id=>document.getElementById(id);
const state={worker:null,id:0,pending:new Map(),ready:false,busy:false,scan:null,particles:null,view:'processed',tool:null,points:[],angle:-.7,tilt:.5,drag:null,rect:null};
const fmt=(v,d=3)=>v==null||!Number.isFinite(Number(v))?'—':Math.abs(v)>=1e5||Math.abs(v)>0&&Math.abs(v)<.001?Number(v).toExponential(2):Number(v).toLocaleString(undefined,{maximumFractionDigits:d});
function text(tag,value,cls){const el=document.createElement(tag);el.textContent=value;if(cls)el.className=cls;return el;}
function notice(message){$('lab-error').textContent=message||'';$('lab-error').hidden=!message;}
function clearScan(){state.scan=null;state.particles=null;state.view='processed';state.tool=null;state.points=[];$('map-empty').hidden=false;$('dataset-label').textContent='No scan loaded.';$('calibration').textContent='NO SCAN';$('scan-dimensions').textContent='Calibrated dimensions appear here.';$('channel').replaceChildren(text('option','Import a scan to choose a channel'));$('history').replaceChildren(text('li','Processing history appears here.'));$('roughness-metrics').replaceChildren(text('p','Load a surface to see roughness metrics.','control-note'));$('surface-quality').textContent='';$('map-min').textContent='—';$('map-max').textContent='—';clearResults();syncView();drawMap();}
function controls(){
  const ready=state.ready&&!state.busy,loaded=ready&&!!state.scan;
  $('start-engine').disabled=state.busy;$('start-engine').hidden=state.ready;$('cancel').hidden=!state.busy;
  for(const id of ['scan-file','load-demo'])$(id).disabled=!ready;
  for(const id of ['channel','reset','export-json','export-matrix','export-image','profile-tool','crop-tool','clear-tool','coordinate-profile','coordinate-crop','analyze-spectrum'])$(id).disabled=!loaded;
  document.querySelectorAll('[data-operation]').forEach(el=>el.disabled=!loaded);
  $('analyze-particles').disabled=!loaded||!state.scan.height_metrology;
  $('export-particles').disabled=!ready||!state.particles;
  $('undo').disabled=!loaded||!state.scan.can_undo;
  document.querySelectorAll('[data-view]').forEach(el=>el.disabled=state.busy);
}
function makeWorker(){
  if(state.scan)clearScan();
  const worker=new Worker(new URL('./afm-worker.mjs',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{if(data.label){$('engine-status').textContent=data.label;return;}const pending=state.pending.get(data.id);if(!pending)return;state.pending.delete(data.id);clearTimeout(pending.timer);data.error?pending.reject(Error(data.error)):pending.resolve(data.result);};
  worker.onerror=()=>{if(state.worker!==worker)return;worker.terminate();state.worker=null;state.ready=false;clearScan();for(const pending of state.pending.values()){clearTimeout(pending.timer);pending.reject(Error('The analysis worker stopped. Relaunch the lab.'));}state.pending.clear();};
  state.worker=worker;
}
async function request(payload,bytes){
  if(state.busy)throw Error('Wait for the current analysis to finish.');
  if(!state.worker)makeWorker();state.busy=true;notice('');controls();
  try{return await new Promise((resolve,reject)=>{const id=++state.id,timer=setTimeout(()=>{state.worker?.terminate();state.worker=null;state.ready=false;clearScan();for(const job of state.pending.values()){clearTimeout(job.timer);job.reject(Error('The analysis timed out. Relaunch the lab.'));}state.pending.clear();},180000);state.pending.set(id,{resolve,reject,timer});state.worker.postMessage({id,payload,...(bytes?{bytes}:{})},bytes?[bytes]:[]);});}
  finally{state.busy=false;controls();}
}
function handle(fn){return async()=>{try{await fn();}catch(error){notice(error.message);$('engine-status').textContent=state.ready?'Ready · your previous scan is preserved':'Launch the lab to start again';}};}
async function launch(){let result;try{result=await request({action:'catalog'});}catch(error){state.worker?.terminate();state.worker=null;state.ready=false;throw error;}state.ready=true;controls();$('engine-status').textContent='Ready · all calculations run on your device';$('engine-version').textContent=`AFM core ${result.engine.version} · NumPy ${result.engine.numpy} · SciPy ${result.engine.scipy} · scikit-image ${result.engine.skimage}. Desktop batch, Excel and 600 DPI figures remain available in the desktop edition.`;}
$('start-engine').addEventListener('click',handle(launch));
$('cancel').addEventListener('click',()=>{
  state.worker?.terminate();state.worker=null;state.ready=false;for(const pending of state.pending.values()){clearTimeout(pending.timer);pending.reject(Error('Stopped. Unsaved analysis was cleared.'));}state.pending.clear();state.busy=false;clearScan();controls();$('engine-status').textContent='Analysis cleared · launch the lab to begin again';
});
function metrics(container,items){container.replaceChildren(...items.map(([label,value,unit])=>{const card=text('div','','afm-metric');card.append(text('strong',fmt(value)),text('small',label+(unit?' / '+unit:'')));return card;}));}
function clearResults(){state.particles=null;$('particle-results').hidden=true;$('particle-empty').hidden=false;$('particle-summary').replaceChildren();$('spectrum-results').hidden=true;$('profile-result').hidden=true;}
function showScan(scan,clear=true){
  state.scan=scan;state.points=[];state.tool=null;if(clear)clearResults();
  $('map-empty').hidden=true;$('dataset-label').textContent=scan.source;
  $('channel').replaceChildren(...scan.channels.map((channel,index)=>{const option=text('option',channel.label);option.value=index;return option;}));$('channel').value=scan.channel_index;
  $('calibration').textContent=scan.calibration;$('calibration').dataset.state=scan.calibration;$('calibration').title=scan.calibration_note;
  $('scan-dimensions').textContent=`${scan.cols} × ${scan.rows} px · ${fmt(scan.x_um)} × ${fmt(scan.y_um)} µm · pitch ${fmt(scan.pitch_x_nm)} / ${fmt(scan.pitch_y_nm)} nm`;
  for(const [id,value]of [['coord-x0',scan.x_um*.25],['coord-y0',scan.y_um*.25],['coord-x1',scan.x_um*.75],['coord-y1',scan.y_um*.75]])$(id).value=value;
  $('history').replaceChildren(...scan.steps.map(step=>text('li',step)));
  const r=scan.roughness;metrics($('roughness-metrics'),[['Sa',r.Sa,scan.height_unit],['Sq',r.Sq,scan.height_unit],['Sz',r.Sz,scan.height_unit],['Ssk',r.Ssk,''],['Sku',r.Sku,''],['Sp',r.Sp,scan.height_unit],['Sv',r.Sv,scan.height_unit],['Mean',r.Mean,scan.height_unit]]);
  $('surface-quality').textContent=`Finite observations: ${fmt(scan.finite_fraction*100,2)}%. ${scan.calibration_note}`;
  $('map-hint').textContent=`Analysis uses all ${scan.rows*scan.cols} pixels. Map preview: ${scan.grid[0].length} × ${scan.grid.length} sampled pixels. ${state.view==='surface'?'Drag to rotate. Vertical proportions are amplified for display.':'Hover to inspect the preview; select two points for a profile or crop.'}`;
  $('engine-status').textContent='Scan ready · calculations and exports stay on your device';drawMap();syncView();controls();
}
$('load-demo').addEventListener('click',handle(async()=>{state.view='processed';showScan(await request({action:'demo'}));}));
$('scan-file').addEventListener('change',handle(async()=>{const file=$('scan-file').files[0];if(!file)return;if(file.size>33554432)throw Error('Choose a scan no larger than 32 MB.');const bytes=await file.arrayBuffer();const scan=await request({action:'import',fileName:file.name,matrixUnit:$('matrix-unit').value,xUm:Number($('matrix-x').value),yUm:Number($('matrix-y').value)},bytes);state.view='processed';showScan(scan);$('scan-file').value='';}));
$('channel').addEventListener('change',handle(async()=>{state.view='processed';showScan(await request({action:'channel',index:Number($('channel').value)}));}));
document.querySelectorAll('[data-operation]').forEach(button=>button.addEventListener('click',handle(async()=>{const scan=await request({action:'process',operation:button.dataset.operation,sigma:Number($('sigma').value),size:Number($('median-size').value),degree:1});state.view='processed';showScan(scan);})));
for(const action of ['undo','reset'])$(action).addEventListener('click',handle(async()=>{state.view='processed';showScan(await request({action}));}));
function syncView(){document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-selected',String(button.dataset.view===state.view)));$('vertical-control').hidden=state.view!=='surface';$('surface-map').classList.toggle('surface-view',state.view==='surface');$('surface-map').classList.toggle('selecting',!!state.tool);for(const mode of ['profile','crop'])$(mode+'-tool').setAttribute('aria-pressed',String(state.tool===mode));}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',handle(async()=>{if(!state.scan)return;const view=button.dataset.view;const scan=await request({action:'view',view:view==='surface'?'processed':view});state.view=view;showScan(scan,false);})));
for(const mode of ['profile','crop'])$(mode+'-tool').addEventListener('click',handle(async()=>{if(state.view!=='processed'){state.view='processed';showScan(await request({action:'view',view:'processed'}),false);}state.tool=mode;state.points=[];syncView();$('map-hint').textContent=mode==='profile'?'Select two points on the processed surface.':'Select two corners of a crop. The full-resolution raster will be cropped.';drawMap();}));
$('clear-tool').addEventListener('click',()=>{state.tool=null;state.points=[];syncView();drawMap();});
document.addEventListener('keydown',event=>{if(event.key==='Escape')$('clear-tool').click();});
for(const mode of ['profile','crop'])$('coordinate-'+mode).addEventListener('click',handle(async()=>{const s=state.scan,points=[[Number($('coord-x0').value)/s.x_um,Number($('coord-y0').value)/s.y_um],[Number($('coord-x1').value)/s.x_um,Number($('coord-y1').value)/s.y_um]];if(points.flat().some(v=>!Number.isFinite(v)||v<0||v>1))throw Error('Coordinates must lie inside the current calibrated field.');if(mode==='profile'){const result=await request({action:'profile',points});plotLine($('profile-plot'),result.distance_um,result.height,'Distance / µm','Height / '+s.height_unit);$('profile-result').hidden=false;}else{state.view='processed';showScan(await request({action:'crop',box:points.flat()}));}}));
const palette=[[24,51,90],[70,109,197],[153,116,222],[234,161,195],[255,240,207]];
function rgb(t){t=Math.max(0,Math.min(1,t))*(palette.length-1);const i=Math.min(palette.length-2,Math.floor(t)),a=t-i;return palette[i].map((v,j)=>Math.round(v+(palette[i+1][j]-v)*a));}
function drawMap(){
  const canvas=$('surface-map'),w=canvas.clientWidth,h=canvas.clientHeight,ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);if(!state.scan||!w)return;
  const grid=state.scan.grid,rows=grid.length,cols=grid[0].length;let min=Infinity,max=-Infinity;for(const row of grid)for(const value of row)if(value!=null&&Number.isFinite(value)){min=Math.min(min,value);max=Math.max(max,value);}if(!Number.isFinite(min))return;const span=max-min||1;
  $('map-min').textContent=fmt(min)+' '+state.scan.height_unit;$('map-max').textContent=fmt(max)+' '+state.scan.height_unit;
  if(state.view==='surface'){
    state.rect=null;const stride=Math.max(1,Math.ceil(Math.max(rows,cols)/44)),scale=Math.min(w*.30,h*.45),vertical=Number($('vertical-scale').value),angle=state.angle,tilt=state.tilt,xy=state.scan.y_um/state.scan.x_um;
    const project=(r,c)=>{const x=c/(cols-1)-.5,y=(r/(rows-1)-.5)*Math.min(3,xy),z=grid[r][c];const u=x*Math.cos(angle)-y*Math.sin(angle),v=x*Math.sin(angle)+y*Math.cos(angle);return [w*.5+u*scale*1.7,h*.57+v*scale*tilt-(z-min)/span*scale*vertical,v];};
    const faces=[];for(let r=0;r<rows-1;r+=stride)for(let c=0;c<cols-1;c+=stride){const rr=Math.min(rows-1,r+stride),cc=Math.min(cols-1,c+stride),values=[grid[r][c],grid[r][cc],grid[rr][cc],grid[rr][c]];if(values.some(v=>v==null))continue;const points=[project(r,c),project(r,cc),project(rr,cc),project(rr,c)];faces.push({points,value:values.reduce((a,b)=>a+b)/4,depth:points.reduce((a,b)=>a+b[2],0)});}faces.sort((a,b)=>a.depth-b.depth);
    for(const face of faces){ctx.beginPath();face.points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=`rgb(${rgb((face.value-min)/span)})`;ctx.fill();ctx.strokeStyle='#091c3044';ctx.lineWidth=.45;ctx.stroke();}
    ctx.font='10px monospace';ctx.fillStyle='#90afc3';ctx.textAlign='center';ctx.fillText(`${fmt(state.scan.x_um)} × ${fmt(state.scan.y_um)} µm · height in ${state.scan.height_unit}`,w*.5,h-25);return;
  }
  const availableW=w-100,availableH=h-75,aspect=state.scan.x_um/state.scan.y_um;let mw=availableW,mh=mw/aspect;if(mh>availableH){mh=availableH;mw=mh*aspect;}const left=(w-mw)/2+8,top=(h-mh)/2-5;state.rect={left,top,width:mw,height:mh};
  const tile=document.createElement('canvas');tile.width=cols;tile.height=rows;const tctx=tile.getContext('2d'),image=tctx.createImageData(cols,rows);
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const value=grid[r][c],offset=((rows-1-r)*cols+c)*4,color=value==null?[40,51,66]:rgb((value-min)/span);image.data.set([...color,255],offset);}tctx.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(tile,left,top,mw,mh);
  if(state.particles&&state.view==='processed'){const labels=state.particles.labels;ctx.fillStyle='#ecfdff55';for(let r=0;r<labels.length;r++)for(let c=0;c<labels[0].length;c++){const label=labels[r][c];if(!label)continue;if(r===0||c===0||r===labels.length-1||c===labels[0].length-1||labels[r-1][c]!==label||labels[r+1][c]!==label||labels[r][c-1]!==label||labels[r][c+1]!==label)ctx.fillRect(left+c/labels[0].length*mw,top+(1-(r+1)/labels.length)*mh,mw/labels[0].length+1,mh/labels.length+1);}}
  ctx.strokeStyle='#9bbed344';ctx.lineWidth=1;ctx.strokeRect(left,top,mw,mh);ctx.fillStyle='#9eb8cc';ctx.font='10px monospace';ctx.textAlign='center';for(let i=0;i<=4;i++){ctx.fillText(fmt(state.scan.x_um*i/4,2),left+mw*i/4,top+mh+19);ctx.textAlign='right';ctx.fillText(fmt(state.scan.y_um*i/4,2),left-9,top+mh*(1-i/4)+4);ctx.textAlign='center';}ctx.fillText('X / µm',left+mw/2,top+mh+40);ctx.save();ctx.translate(left-40,top+mh/2);ctx.rotate(-Math.PI/2);ctx.fillText('Y / µm',0,0);ctx.restore();
  const points=state.points.map(([x,y])=>[left+x*mw,top+(1-y)*mh]);ctx.strokeStyle='#f3f9ff';ctx.fillStyle='#f3f9ff';ctx.lineWidth=1.5;for(const [x,y]of points){ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();}if(points.length===2){ctx.beginPath();ctx.moveTo(...points[0]);ctx.lineTo(...points[1]);ctx.stroke();}
}
new ResizeObserver(drawMap).observe($('map-container'));
$('vertical-scale').addEventListener('input',drawMap);
function mapPoint(event){const rect=state.rect;if(!rect)return null;const bounds=$('surface-map').getBoundingClientRect(),x=(event.clientX-bounds.left-rect.left)/rect.width,y=1-(event.clientY-bounds.top-rect.top)/rect.height;return x>=0&&x<=1&&y>=0&&y<=1?[x,y]:null;}
$('surface-map').addEventListener('pointermove',event=>{
  if(state.drag&&state.view==='surface'){state.angle+=(event.clientX-state.drag.x)*.006;state.tilt=Math.max(.2,Math.min(.9,state.tilt+(event.clientY-state.drag.y)*.003));state.drag={x:event.clientX,y:event.clientY};drawMap();return;}
  const point=mapPoint(event),tip=$('map-tooltip');if(!point||!state.scan){tip.hidden=true;return;}const [x,y]=point,grid=state.scan.grid,value=grid[Math.min(grid.length-1,Math.floor(y*grid.length))][Math.min(grid[0].length-1,Math.floor(x*grid[0].length))];tip.textContent=`${fmt(x*state.scan.x_um)} / ${fmt(y*state.scan.y_um)} µm · ${fmt(value)} ${state.scan.height_unit}`;const bounds=$('surface-map').getBoundingClientRect();tip.style.left=Math.min(bounds.width-220,Math.max(8,event.clientX-bounds.left+12))+'px';tip.style.top=Math.max(8,event.clientY-bounds.top-35)+'px';tip.hidden=false;
});
$('surface-map').addEventListener('pointerleave',()=>{$('map-tooltip').hidden=true;});
$('surface-map').addEventListener('pointerdown',event=>{if(state.view==='surface'){state.drag={x:event.clientX,y:event.clientY};$('surface-map').setPointerCapture(event.pointerId);}});
$('surface-map').addEventListener('pointerup',()=>{state.drag=null;});
$('surface-map').addEventListener('click',async event=>{
  if(!state.tool||state.busy)return;const point=mapPoint(event);if(!point)return;state.points.push(point);drawMap();if(state.points.length<2)return;const mode=state.tool,points=state.points.slice();state.tool=null;syncView();
  try{if(mode==='profile'){const result=await request({action:'profile',points});plotLine($('profile-plot'),result.distance_um,result.height,'Distance / µm','Height / '+state.scan.height_unit);$('profile-result').hidden=false;}
    else{const box=[Math.min(points[0][0],points[1][0]),Math.min(points[0][1],points[1][1]),Math.max(points[0][0],points[1][0]),Math.max(points[0][1],points[1][1])];showScan(await request({action:'crop',box}));}}
  catch(error){notice(error.message);state.points=[];drawMap();}
});
function plotLine(host,x,y,xlabel,ylabel,log=false){
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 560 215');svg.setAttribute('role','img');svg.setAttribute('aria-label',ylabel+' against '+xlabel);
  const rows=x.map((v,i)=>[v,y[i]]).filter(row=>row.every(v=>v!=null&&Number.isFinite(v))&&(!log||row.every(v=>v>0))).map(([a,b])=>log?[Math.log10(a),Math.log10(b)]:[a,b]);if(rows.length<2){host.replaceChildren(text('p','Not enough finite values to draw this plot.','control-note'));return;}
  const xmin=Math.min(...rows.map(r=>r[0])),xmax=Math.max(...rows.map(r=>r[0])),ymin=Math.min(...rows.map(r=>r[1])),ymax=Math.max(...rows.map(r=>r[1]));
  const X=v=>62+(v-xmin)/(xmax-xmin||1)*470,Y=v=>165-(v-ymin)/(ymax-ymin||1)*135;
  const add=(tag,attrs,value)=>{const el=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))el.setAttribute(k,v);if(value)el.textContent=value;svg.append(el);return el;};
  for(let i=0;i<=4;i++){const a=xmin+(xmax-xmin)*i/4,b=ymin+(ymax-ymin)*i/4;add('path',{class:'grid',d:`M${X(a)} 30V165 M62 ${Y(b)}H532`});add('text',{x:X(a),y:184,'text-anchor':'middle'},fmt(log?10**a:a,2));add('text',{x:55,y:Y(b)+3,'text-anchor':'end'},fmt(log?10**b:b,2));}
  add('path',{class:'trace',d:rows.map(([a,b],i)=>`${i?'L':'M'}${X(a).toFixed(2)} ${Y(b).toFixed(2)}`).join(' ')});add('text',{x:300,y:206,'text-anchor':'middle'},xlabel);add('text',{x:17,y:95,transform:'rotate(-90 17 95)','text-anchor':'middle'},ylabel);host.replaceChildren(svg);
}
$('analyze-particles').addEventListener('click',handle(async()=>{
  const result=await request({action:'morphology',method:$('threshold-method').value,direction:$('direction').value,threshold:Number($('threshold').value),kSigma:Number($('k-sigma').value),smooth:Number($('morph-sigma').value),minArea:Number($('min-area').value),pixelsAcross:Number($('pixels-across').value),plane:$('morph-plane').checked,watershed:$('watershed').checked,clearBorder:$('clear-border').checked});state.particles=result;
  const valid=result.records.filter(r=>r.size_qc_pass),mean=(rows,key)=>rows.length?rows.reduce((a,b)=>a+b[key],0)/rows.length:null;
  metrics($('particle-summary'),[['Detected',result.records.length,'particles'],['Sampling-QC valid',valid.length,'particles'],['Mean apparent diameter',mean(valid,'equivalent_diameter'),'nm'],['Mean P95 height',mean(valid,'p95_height'),'nm']]);
  $('particle-rows').replaceChildren(...result.records.slice(0,1000).map(record=>{const row=document.createElement('tr');for(const value of [record.particle_id,fmt(record.equivalent_diameter),fmt(record.p95_height),fmt(record.area),fmt(record.volume),record.size_qc_pass?'Pass':'Flagged'])row.append(text('td',String(value)));row.lastChild.title=record.size_qc_note;if(!record.size_qc_pass)row.lastChild.className='qc-fail';return row;}));
  $('particle-empty').hidden=true;$('particle-results').hidden=false;$('particle-note').textContent=`Threshold: ${fmt(result.threshold_nm)} nm. All detected particles are retained in the export; summary means use sampling-QC valid particles. ${result.records.length>1000?'Table preview shows the first 1000.':''}`;drawMap();controls();$('engine-status').textContent=`Particle analysis complete · ${result.records.length} particles detected`;
}));
$('analyze-spectrum').addEventListener('click',handle(async()=>{const result=await request({action:'spectrum'});plotLine($('psd-plot'),result.frequency,result.psd,'Frequency / cycles µm⁻¹',result.psd_unit,true);plotLine($('acf-plot'),result.acf_distance_um,result.acf,'X displacement / µm','Normalized ACF');$('spectrum-results').hidden=false;$('spectrum-note').textContent='PSD is an annular mean of the 2D spectral density, shown on log axes. ACF is the positive X section, not a radial correlation-length estimate.';$('engine-status').textContent='Spatial analysis complete · results stay on your device';}));
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);}
const csvValue=value=>{const s=value==null?'':String(value);return /[",\n\r]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
const stem=()=>String(state.scan?.source||'afm').replace(/[^a-zA-Z0-9_-]/g,'_').slice(0,90);
$('export-json').addEventListener('click',handle(async()=>download(JSON.stringify(await request({action:'export'}),null,2),stem()+'_analysis.json','application/json')));
$('export-matrix').addEventListener('click',handle(async()=>{const result=await request({action:'export',matrix:true});download(`# Processed height matrix in ${state.scan.height_unit}; X=${result.x_um} µm; Y=${result.y_um} µm; pixel edges convention\n`+result.matrix.map(row=>row.map(v=>v==null?'NaN':v).join(',')).join('\n'),stem()+'_surface.csv','text/csv');}));
$('export-particles').addEventListener('click',()=>{if(!state.particles)return;const fields=['particle_id','centroid_x','centroid_y','equivalent_diameter','equivalent_diameter_low','equivalent_diameter_high','p95_height','max_height','area','volume','circularity','pixels_across','size_qc_pass','size_qc_note','touches_border','touches_missing_data'];download('# Length: nm; area: nm²; volume: nm³; lateral sizes are apparent\n'+fields.join(',')+'\n'+state.particles.records.map(record=>fields.map(key=>csvValue(record[key])).join(',')).join('\n'),stem()+'_particles.csv','text/csv');});
$('export-image').addEventListener('click',()=>{if(!state.scan)return;const source=$('surface-map'),scale=source.width/source.clientWidth,image=document.createElement('canvas');image.width=source.width;image.height=source.height+Math.round(105*scale);const ctx=image.getContext('2d');ctx.fillStyle='#091321';ctx.fillRect(0,0,image.width,image.height);ctx.drawImage(source,0,0);ctx.scale(scale,scale);ctx.fillStyle='#b8d6e5';ctx.font='11px monospace';const y=source.clientHeight,w=source.clientWidth-36;ctx.fillText(`Afruz AFM Studio · core v0.4.0 · ${state.view} view`,18,y+23,w);ctx.fillText(`${state.scan.source} · ${state.scan.cols} × ${state.scan.rows} px`,18,y+44,w);ctx.fillText(`Field: ${fmt(state.scan.x_um)} × ${fmt(state.scan.y_um)} µm · Height: ${$('map-min').textContent} → ${$('map-max').textContent}`,18,y+65,w);ctx.fillStyle='#7897b0';ctx.font='9px monospace';ctx.fillText(`Source SHA-256: ${state.scan.source_sha256.slice(0,16)} · ${state.view==='surface'?'Vertical proportions amplified':'Sampled map preview; analysis uses full raster'}`,18,y+86,w);image.toBlob(blob=>{if(blob)download(blob,stem()+'_map.png','image/png');});});
$('copy-citation').addEventListener('click',handle(async()=>{await navigator.clipboard.writeText('AliAfruz. (2026). Afruz AFM Studio: Orbit Workspace (v0.4.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23348914');$('engine-status').textContent='Citation copied · thank you for crediting the software';}));
controls();drawMap();
