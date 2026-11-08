import fs from 'node:fs/promises';
import {Presentation} from 'file:///C:/Program%20Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {build} from './build_deck.mjs';
const p=Presentation.create({slideSize:{width:1280,height:720}});
await build({presentation:p,input:{from:1,to:23,local:true}});
const dir='D:/Dev/tilelang/ppt_workspace/visual-v2-renders';
await fs.mkdir(dir,{recursive:true});
const selected=process.argv.slice(2).map(Number);
for(let n=1;n<=23;n++){
 if(selected.length&&!selected.includes(n))continue;
 const blob=await p.export({slide:p.slides.getItem(n-1),format:'png',scale:1.25});
 await fs.writeFile(`${dir}/slide-${String(n).padStart(2,'0')}.png`,new Uint8Array(await blob.arrayBuffer()));
 console.log('Rendered '+n);
}
