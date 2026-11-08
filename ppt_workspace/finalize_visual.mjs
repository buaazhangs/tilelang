import fs from 'node:fs/promises';
import path from 'node:path';
import {finalizePresentation} from 'file:///C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.22356/skills/presentations/container_tools/artifact_tool_utils.mjs';
const base='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1104.22356/skills/presentations/container_tools';
const runtime='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt/dependencies';
const result=await finalizePresentation({
 workspaceDir:'D:/Dev/tilelang',
 candidatePath:'D:/Dev/tilelang/ppt_workspace/visual-animated-candidate.pptx',
 finalPath:'D:/Dev/tilelang/PPT交付/TileLang_昇腾模调度_视觉精修版.pptx',
 pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:base+'/inspect_presentation_package_integrity.py',
 layoutValidatorPath:base+'/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:23,
 requiredNativeChartOwnerSlides:[19],
 materializeLiteralChartWorkbooks:false,
 fontPolicy:{basis:'design',families:['Microsoft YaHei','微软雅黑','Consolas'],scriptFonts:{ea:'Microsoft YaHei'}},
 verifyArtifactToolImport:true,
 receiptPath:'D:/Dev/tilelang/ppt_workspace/visual-final-validation-v2.json'
});
console.log(JSON.stringify(result,null,2));
