// Uploaded files remain data in this worker's memory-only filesystem.
import {loadPyodide} from '../eis/runtime/pyodide.mjs';
let engine, request, bootPromise, running=false;
const actions=new Set(['catalog','demo','import','channel','process','view','undo','reset','crop','profile','spectrum','morphology','export']);
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
async function boot(){
  postMessage({label:'Loading the scientific runtime on your device…'});
  const runtime=new URL('../eis/runtime/',import.meta.url).href;
  engine=await loadPyodide({indexURL:runtime,lockFileURL:new URL('../eis/runtime/packages.txt',import.meta.url).href,stdLibURL:new URL('../eis/runtime/stdlib.bin',import.meta.url).href,stdout:()=>{},stderr:()=>{}});
  postMessage({label:'Loading NumPy, SciPy and particle-analysis tools…'});
  await engine.loadPackage(['numpy','scipy','scikit-image']);
  const response=await fetch(new URL('./source-manifest.txt',import.meta.url));
  if(!response.ok)throw Error('AFM source manifest could not load.');
  const manifest=await response.json();
  if(manifest.commit!=='08454897c122f60b617d6909029fa94e9cb7ce2a'||manifest.files?.length!==5)throw Error('AFM source manifest does not identify the archived release.');
  const allow=new Set(['afm_analysis.py','morphology_analysis.py','nid_reader.py','reporting.py','examples/synthetic_afm.py']);
  for(const file of manifest.files){
    if(!allow.delete(file.path)||!/^[a-f0-9]{64}$/.test(file.sha256)||file.bytes>100000)throw Error('Unexpected AFM source module.');
    const result=await fetch(new URL(`./source/${file.path}.txt`,import.meta.url));
    if(!result.ok)throw Error('AFM scientific module could not load.');
    const bytes=await result.arrayBuffer();
    if(bytes.byteLength!==file.bytes||await hash(bytes)!==file.sha256)throw Error('AFM source verification failed.');
    const location='/home/pyodide/'+file.path;
    engine.FS.mkdirTree(location.slice(0,location.lastIndexOf('/')));engine.FS.writeFile(location,new Uint8Array(bytes));
  }
  const bridge=await fetch(new URL('./bridge.py.txt',import.meta.url));
  if(!bridge.ok)throw Error('AFM browser adapter could not load.');
  await engine.runPythonAsync(await bridge.text());request=engine.globals.get('web_request');
}
self.onmessage=async({data})=>{
  const id=data?.id;let claimed=false;
  try{
    if(!Number.isSafeInteger(id)||!actions.has(data.payload?.action)||JSON.stringify(data.payload).length>8192)throw Error('Unsupported AFM request.');
    if(running)throw Error('Wait for the current analysis to finish.');
    running=true;claimed=true;bootPromise??=boot();await bootPromise;
    if(data.payload.action==='import'){
      if(!(data.bytes instanceof ArrayBuffer)||data.bytes.byteLength===0||data.bytes.byteLength>33554432)throw Error('Choose a scan up to 32 MB.');
      engine.FS.writeFile('/tmp/afm-import.data',new Uint8Array(data.bytes));
    }
    try{postMessage({id,result:JSON.parse(request(JSON.stringify(data.payload)))});}
    finally{if(data.payload.action==='import')engine.FS.unlink('/tmp/afm-import.data');}
  }catch(error){postMessage({id,error:String(error.message||error).slice(-1400)});}
  finally{if(claimed)running=false;}
};
