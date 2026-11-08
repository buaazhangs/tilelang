import fs from 'node:fs/promises';
import {Presentation,PresentationFile,FileBlob} from 'file:///C:/Program%20Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {build,ROOT} from './build.mjs';
const mode=process.argv[2]||'build';
let p;
if(mode==='source')p=await PresentationFile.importPptx(await FileBlob.load(ROOT+'/source-user.pptx'));
else if(mode==='endstate')p=await PresentationFile.importPptx(await FileBlob.load(ROOT+'/endstate.pptx'));
else if(mode==='candidate')p=await PresentationFile.importPptx(await FileBlob.load(ROOT+'/candidate.pptx'));
else {p=Presentation.create({slideSize:{width:1280,height:720}});await build({presentation:p,input:{from:1,to:24,local:true}});await(await PresentationFile.exportPptx(p)).save(ROOT+'/static.pptx')}
const dir=ROOT+'/'+mode+'-renders';await fs.mkdir(dir,{recursive:true});
for(let n=1;n<=p.slides.items.length;n++) {const b=await p.export({slide:p.slides.getItem(n-1),format:'png',scale:1.25});await fs.writeFile(`${dir}/slide-${String(n).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(mode+' rendered '+p.slides.items.length+' slides');
