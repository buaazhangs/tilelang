import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Program%20Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/Dev/tilelang/ppt_workspace/visual-animated-candidate.pptx'));
const dir='D:/Dev/tilelang/ppt_workspace/visual-v2-final-renders';
await fs.mkdir(dir,{recursive:true});
for(let n=1;n<=23;n++){
 const blob=await p.export({slide:p.slides.getItem(n-1),format:'png',scale:1.25});
 await fs.writeFile(`${dir}/slide-${String(n).padStart(2,'0')}.png`,new Uint8Array(await blob.arrayBuffer()));
}
console.log('Rendered all 23 slides from the exact animated candidate.');
