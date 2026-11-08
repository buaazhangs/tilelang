import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from 'file:///C:/Program%20Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import { build } from './build_deck.mjs';
// The live session returns a download URL rather than local bytes, and its
// sandbox forbids the package's node:process import. Use the same reviewed
// authoring callback in the bundled local Artifact Tool for PPTX delivery.
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
await build({presentation,input:{from:1,to:23,local:true}});
await (await PresentationFile.exportPptx(presentation)).save('D:/Dev/tilelang/ppt_workspace/static-draft.pptx');
console.log('Exported 23 editable slides');
