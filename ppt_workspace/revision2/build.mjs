import fs from 'node:fs/promises';
export const ROOT='D:/Dev/tilelang/ppt_workspace/revision2';
const C={red:'#B3202B',redL:'#F9E9EB',ink:'#242933',muted:'#66707B',line:'#DCE1E6',blue:'#326DA8',blueL:'#EAF1F8',orange:'#C16B24',orangeL:'#F9EEE4',green:'#28795F',greenL:'#E8F3EE',purple:'#7352A0',purpleL:'#F0EAF7',white:'#FFFFFF'};
export const titles=['Tilelang — 支持昇腾算子高性能运行','硬件利用率','FA在昇腾硬件的数据流程','串行执行限制硬件利用率','在等待期间执行其他批次任务','昇腾950执行模型','GPU的两种warp组织方式','见缝插针','昇腾950执行模型','提前调整执行顺序，提高硬件利用率','不同任务耗时，让流水编排更复杂','TileLang Autoschedule — 编译器使能流水排布','TileLang的职责','从硬件利用率到启动间隔II','改变发射顺序，需要决定什么','任务的时间模型与调度变量','数据依赖约束：生产者–消费者','数据依赖约束：同任务存储','硬件使用约束','建模总结','固定II，搜索可行的流水方案','调度结果转化为指令与多缓冲','硬件利用率能提升多少','谢谢，欢迎提问'];
export const clicks=[0,0,4,3,3,0,3,4,6,4,3,0,3,4,3,3,3,3,3,3,4,4,3,0];
const chapter=n=>n<6?0:n<12?1:2;
export async function build({presentation,input}){
 let speech='';try{speech=await fs.readFile(ROOT+'/speech_draft.md','utf8')}catch{}
 const notes={};for(const m of speech.matchAll(/^## P(\d{2})[^\n]*\n([\s\S]*?)(?=^## |$(?![\s\S]))/gm))notes[+m[1]]=m[2];
 const out=[];
 for(let n=input.from;n<=input.to;n++){
 const s=n<=presentation.slides.items.length?presentation.slides.getItem(n-1):presentation.slides.add({width:1280,height:720});s.shapes.deleteAll();for(const im of [...s.images.items])im.delete();for(const ch of [...s.charts.items])presentation.delete(ch.toSnapshot().aid);s.background.fill=C.white;s.hidden=false;
 let seq=0;const animations=[];const items=[];
 const nm=(k,key)=>`R2_P${String(n).padStart(2,'0')}_C${String(k).padStart(2,'0')}_${++seq}_${key}`;
 function sh(x,y,w,h,fill='none',k=0,key='shape',stroke='none',geometry='rect') {const e=s.shapes.add({name:nm(k,key),geometry,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:stroke==='none'?0:1.3}});items.push({name:e.name,k,kind:key});if(k)animations.push({name:e.name,click:k,effect:'fade'});return e;}
 function tx(t,x,y,w,h=42,size=27,color=C.ink,k=0,bold=false,align='left'){const e=sh(x,y,w,h,'none',k,'text');e.text=t;e.text.style={typeface:'Microsoft YaHei',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'square'};e.text.insets=0;return e;}
 function bx(t,x,y,w,h,color=C.blue,k=0,light=false,size=24){const e=sh(x,y,w,h,light?C.white:color,k,'task',color,'roundRect');e.text=t;e.text.style={typeface:'Microsoft YaHei',fontSize:size,color:light?color:C.white,bold:false,alignment:'center',verticalAlignment:'middle',autoFit:'none'};e.text.insets=5;return e;}
 function ln(x,y,x2,y2,color=C.line,k=0,arrow=false,width=2){const w=Math.abs(x2-x)||1,h=Math.abs(y2-y)||1;const e=s.shapes.add({name:nm(k,'line'),geometry:'custom',position:{left:Math.min(x,x2),top:Math.min(y,y2),width:w,height:h},fill:'none',line:{fill:color,width},customPaths:[{width:w,height:h,commands:[{moveTo:{x:x< x2?0:w,y:y<y2?0:h}},{lineTo:{x:x<x2?w:0,y:y<y2?h:0}}]}]});items.push({name:e.name,k,kind:'line'});if(k)animations.push({name:e.name,click:k,effect:'fade'});if(arrow){const head=sh(x2-5,y2-6,10,12,color,k,'arrowhead','none','triangle');head.position={...head.position,rotation:Math.atan2(y2-y,x2-x)*180/Math.PI+90};}return e;}
 function move(e,click,dx,dy,duration=.65,options={}){animations.push({name:e.name,click,effect:'motion',dx,dy,duration,...options});}
 function sub(t,x,y,w,k=0){return tx(t,x,y,w,42,29,C.red,k,true);}
 function f(t,x,y,w,k=0,size=32,color=C.ink){return tx(t,x,y,w,55,size,color,k,true);}
 function next(t){ln(64,641,1216,641,C.line);tx(t,64,651,1145,40,24,C.red,0);}
 function lane(label,y,x=190,w=1000,k=0){tx(label,64,y,120,40,23,C.muted,k);ln(x,y+44,x+w,y+44,C.line,k);}
 function task(t,x,y,w,c,k=0){return bx(t,x,y,w,39,c,k,false,21);}
 if(![1,2,6,12,24].includes(n)){
 tx(titles[n-1],64,38,1160,74,n===17||n===18?38:42,C.red,0,true);ln(64,125,1216,125,C.red,0,false,2);
 tx(['硬件利用率','昇腾950执行模型','TileLang Autoschedule'][chapter(n)],64,691,980,20,15,C.muted);tx(String(n).padStart(2,'0'),1160,684,56,30,18,C.muted,0,false,'right');
 }
 switch(n){
 case 1:
 tx('Tilelang',78,158,1100,94,68,C.red,0,true);tx('支持昇腾算子高性能运行',78,260,1125,100,60,C.red,0,true);ln(80,403,310,403,C.red,0,false,5);tx('TileLang Ascend AutoSchedule 的模调度设计',80,437,1120,58,32,C.muted);tx('硬件利用率   /   昇腾950执行模型   /   编译器使能流水排布',80,606,1120,40,25,C.muted);break;
 case 2:case 6:case 12:{const ch={2:1,6:2,12:3}[n];tx('0'+ch,70,106,340,190,150,C.redL,0,true);tx(ch===3?'TileLang Autoschedule':titles[n-1],76,313,1140,84,ch===3?57:64,C.red,0,true);if(ch===3)tx('编译器使能流水排布',80,405,1120,70,45,C.red,0,true);ln(80,519,310,519,C.red,0,false,4);tx(['为什么算力没有被充分利用？','怎样让其他批次填补等待？','怎样把计算流程变成高效流水？'][ch-1],80,550,1120,64,33,C.ink);break;}
 case 3:{
 f('O = softmax(QKᵀ / √d) V',64,145,1160,0,35);
 const im=s.images.add({...(input.local?{blob:new Uint8Array(await fs.readFile('D:/Dev/tilelang/PPT交付/docs2Fguide2Fnpu_3510_hw_arch.png')),contentType:'image/png'}:{path:'D:/Dev/tilelang/PPT交付/docs2Fguide2Fnpu_3510_hw_arch.png'}),fit:'contain',position:{left:54,top:211,width:590,height:420},alt:'用户提供的昇腾950 AI Core架构图'});im.name=nm(0,'arch');
 const flow=[['Q/K/V搬入：GM → L1 / UB','MTE2',C.blue],['Q/K装载：L1 → L0A / L0B','MTE1',C.blue],['QKᵀ：矩阵乘','Cube',C.orange],['S搬出：L0C → UB','FIXPIPE',C.blue],['online softmax：S → P','Vector',C.green],['P回传与P/V装载：UB → L1 → L0','MTE3 / MTE1',C.blue],['PV：矩阵乘','Cube',C.orange],['O搬出与累加：L0C → UB','FIXPIPE / Vector',C.green],['O写回：UB → GM','MTE3',C.blue]];
 flow.forEach(([a,b,c],i)=>{let k=i<2?1:i<4?2:i<6?3:4;bx(a,716,214+i*46,490,38,c,k,true,20);if(i<8)ln(961,253+i*46,961,258+i*46,C.muted,k,true,1.5);});
const targets=[[267,293],[373,271],[473,275],[536,316],[522,544],[420,483],[473,275],[522,544],[420,483]];
targets.forEach(([x,y],i)=>{const k=i<2?1:i<4?2:i<6?3:4;ln(710,233+i*46,x,y,[C.blue,C.blue,C.orange,C.blue,C.green,C.blue,C.orange,C.green,C.blue][i],k,true,1);});
break;}
 case 4:{
 sub('多个批次依次执行，Cube仍存在大量空闲',64,149,1160,0);
const lanes=[0,1,0,2,0,1,0,2],labs=['搬入','QK','搬出','softmax','回传','PV','搬出','累加'];
for(let batch=0;batch<2;batch++){const yy=215+batch*135;tx('批次 '+batch,64,yy-4,110,36,21,C.red,1,true);['搬运','Cube','Vector'].forEach((l,i)=>{tx(l,164,yy+i*41,90,35,20,C.muted,1);ln(260,yy+i*41+37,965,yy+i*41+37,C.line,1)});for(let i=0;i<8;i++)bx(labs[i],260+i*88,yy+lanes[i]*41,84,33,[C.blue,C.orange,C.green][lanes[i]],1,false,18);}
tx('25%',1005,239,210,90,62,C.orange,2,true);tx('Cube忙碌时间占比',981,331,239,63,24,C.ink,2);tx('假设各个任务\n执行时间相同',989,412,230,62,20,C.muted,2);
 tx('950PR：32个Cube核，FP16 Cube峰值 432 TFLOPS',64,501,1150,43,29,C.ink,3);f('432 × 25% = 108 TFLOPS',64,554,630,3,35,C.orange);ln(705,580,802,580,C.red,3,true,3);tx('实际算力利用不足',830,550,380,64,33,C.red,3,true);tx('等时示例：假设各核一致、Cube工作时达到峰值',64,602,1120,30,20,C.muted,3);next('怎样减少等待，让更多批次同时推进？');break;}
 case 5:{
 ['搬运单元','Cube'].forEach((l,i)=>lane(l,223+i*115,210,960));task('K₀',210,223,140,C.blue);task('QK₀',365,338,270,C.orange);const e=task('K₁',920,223,180,C.blue);move(e,1,-540,0);ln(920,202,390,202,C.blue,1,true,2.5);tx('在QK₀执行期间提前搬入K₁',640,310,565,55,27,C.blue,1);
 sub('指令重排',75,457,500,2);tx('把已就绪的其他批次任务提前发射',75,507,530,83,28,C.ink,2);sub('多buffer',700,457,500,3);bx('buffer 0：K₀',700,515,238,58,C.blue,3,true,25);bx('buffer 1：K₁',960,515,238,58,C.blue,3,false,25);next('如何实现？先看硬件的执行模型。');break;}
 case 7:{
 bx('warp scheduler',480,159,320,60,C.red,0,false,29);['Tensor Core','CUDA Core','Load / Store'].forEach((t,i)=>{bx(t,285+i*260,259,235,56,[C.orange,C.green,C.blue][i],0,true,25);ln(640,224,400+i*260,254,C.line,0,true)});
 sub('同质warp',64,370,530,1);sub('warp specialization',700,370,520,2);for(let i=0;i<3;i++){bx('warp '+i,64,437+i*53,115,42,C.blue,1,true,22);tx('搬运 → QK → softmax → PV',197,437+i*53,430,42,23,C.ink,1)}
 [['生产者warp','搬运 / 数据准备',C.blue],['消费者warp 0','矩阵计算',C.orange],['消费者warp 1','向量处理',C.green]].forEach(([a,b,c],i)=>{bx(a,700,437+i*53,205,42,c,2,true,23);tx(b,927,437+i*53,300,42,23,C.ink,2)});tx('warp：拥有寄存器状态的指令流',64,603,1120,36,27,C.red,3,true);next('各个warp如何协同工作？');break;}
 case 8:{
 sub('同质warp：切换到另一个就绪warp',64,153,566);sub('专职warp：选择就绪的角色',700,153,520);for(let side=0;side<2;side++){const x=64+side*636;tx('等待',x,224,92,36,23,C.muted,1);bx(side?'消费者0：等数据':'warp 0：等数据',x+100,220,420,46,C.red,1,true,24);tx('就绪',x,300,92,36,23,C.muted,side?3:2);bx(side?'生产者：搬运下一批':'warp 1：执行QK₁',x+100,297,420,46,side?C.blue:C.orange,side?3:2,false,24);tx('发射',x,398,92,36,23,C.muted,side?3:2);bx('warp scheduler',x+100,394,420,62,C.red,side?3:2,false,27);ln(x+312,350,x+312,388,C.red,side?3:2,true);}
 f('谁就绪，就发射谁',64,504,1150,4,43,C.red);tx('提供足够的就绪工作，才能填补硬件空档',64,567,1150,45,29,C.ink,4);next('昇腾的执行模型怎样组织任务？');break;}
 case 9:{
 sub('Scalar按程序顺序分发指令',64,149,1150);sh(64,209,1150,110,C.white,0,'scalar',C.red);tx('Scalar',78,215,150,30,24,C.red,0,true);
 const arr=[['搬入₀',C.blue,90,440],['QK₀',C.orange,450,440],['softmax₀',C.green,815,440],['PV₀',C.orange,450,505],['QK₁',C.orange,450,570],['softmax₁',C.green,815,505]];
 arr.forEach(([l,c,dx,dy],i)=>{const x=90+i*181;const e=bx(l,x,257,159,45,c,0,false,24);move(e,i+1,dx-x,dy-257,.6)});
 ['搬运单元','Cube','Vector'].forEach((l,i)=>{sh(75+i*360,375,330,245,'none',0,'unit',C.line);tx(l,90+i*360,384,300,42,29,[C.blue,C.orange,C.green][i],0,true);});tx('各单元按既定队列执行，通过同步等待数据就绪',64,650,1150,37,26,C.red,6,true);break;}
 case 10:{
 sub('既定顺序：QK₀ → PV₀ → QK₁',64,153,1150,1);['QK₀','PV₀：等softmax₀','QK₁：已就绪'].forEach((t,i)=>bx(t,64+i*386,218,355,66,i===1?C.red:C.orange,1,i===1,27));tx('Cube队列停在PV₀；后面的QK₁不会自动越过它',64,308,1150,62,31,C.red,2,true);
 sub('编译前调整发射顺序',64,403,1150,3);['QK₀','PV₀','QK₁'].forEach((t,i)=>{const e=bx(t,64+i*386,467,355,66,C.orange,3,false,29);if(i){const dx=i===1?386:-386,dy=i===1?80:-80;move(e,4,dx,0,1.2,{pathPoints:[[0,0],[0,dy],[dx,dy],[dx,0]]});}});const summary=tx('利用softmax₀执行期间推进QK₁',64,570,1150,50,31,C.ink,4);animations.find(a=>a.name===summary.name).delay=1.2;break;}
 case 11:{
 tx('同一时间窗口：B任务耗时 = 4 × A任务耗时',64,155,1160,55,34,C.ink,1,true);lane('单元B',277,240,920,1);task('B₀',240,277,920,C.green,1);lane('单元A',391,240,920,2);for(let i=0;i<4;i++)task('A'+i,240+i*230,391,220,C.orange,2);tx('每个A任务还必须满足数据就绪、buffer可用和资源可用',64,503,1160,62,29,C.ink,3);sub('任务越多，耗时与依赖的组合越复杂',64,579,1160,3);next('TileLang编译后端自动排流水，支持昇腾算子高性能运行。');break;}
 case 13:{
 sub('用户表达tile级别数据计算流程',64,157,590,1);tx('for kv in T.Pipelined(...):\n    qk(...)\n    softmax(...)\n    pack_p(...)\n    pv(...)\n    accumulate_output(...)',80,237,540,300,28,C.ink,1);sub('编译器使能流水排布',733,157,485,2);['分析任务与依赖','估计数据等待与硬件占用','求解阶段、顺序与buffer份数','生成循环并插入同步'].forEach((t,i)=>tx(t,733,242+i*77,480,56,27,C.ink,2));ln(623,383,700,383,C.red,3,true,4);next('TileLang拿到计算流程后，如何建模和分析？');break;}
 case 14:{
 tx('计算量不变：总时间越短，实际算力与MFU越高',64,145,1160,45,32,C.red,1,true);for(let i=0;i<4;i++){tx('批次 '+i,64,255+i*55,125,40,23,C.muted,2);const x=215+i*180;task('QK',x,255+i*55,115,C.orange,2);task('softmax',x+120,255+i*55,140,C.green,2);task('PV',x+265,255+i*55,115,C.orange,2)}ln(215,234,395,234,C.red,3,true,3);tx('II',270,198,100,30,28,C.red,3,true);ln(215,478,755,478,C.red,4,false,2);ln(755,478,1135,478,C.blue,4,false,2);[215,755,1135].forEach(x=>ln(x,470,x,486,C.muted,4));tx('(N − 1) × II',260,488,440,32,26,C.red,4,false,'center');tx('末轮时长',790,488,300,32,26,C.blue,4,false,'center');tx('重叠区域：硬件交错执行不同批次任务',64,531,1160,38,28,C.ink,3);f('总时间 ≈ (N − 1) × II + 末轮时长',64,579,1160,4,36);next('长循环中，减小启动间隔II能直接缩短总时间。');break;}
 case 15:{
 sub('希望的Cube发射顺序：QK₀ → QK₁ → PV₀',64,153,1160);['周期0','周期1','周期2'].forEach((t,i)=>{tx(t,64+i*385,215,350,40,25,C.muted);ln(64+i*385,263,64+i*385,393,C.line)});task('QK₀',92,278,190,C.orange,1);task('QK₁',449,278,155,C.orange,1);task('PV₀',630,278,155,C.orange,2);tx('k：让QK比PV提前一个调度阶段',64,420,1160,52,31,C.red,1,true);tx('r：同一发射周期内，先QK、后PV',64,484,1160,52,31,C.red,2,true);tx('b：QK₁不能覆盖softmax₀仍需读取的QK₀结果',64,548,1160,62,29,C.red,3,true);bx('buffer 0：QK₀',824,277,370,50,C.blue,3,true,25);bx('buffer 1：QK₁',824,341,370,50,C.blue,3,false,25);break;}
 case 16:{
 sub('待求解：调度安排',64,153,540,1);tx('kᵢ  发射阶段\nrᵢ  周期内发射位置\nb  缓冲区版本数',64,215,535,190,32,C.purple,1);sub('已知：任务与硬件信息',700,153,510,2);tx('Lᵢ  数据访问完成等待时间\nHᵢ  硬件资源占用间隔\n任务依赖、buffer大小、片上容量',700,215,520,190,27,C.blue,2);ln(64,439,1216,439,C.line);f('Sᵢ = kᵢ × II + rᵢ',64,477,700,3,46);tx('0 ≤ rᵢ < II',827,487,390,52,32,C.muted,3);tx('第n轮任务i的开始时间：Sᵢ + n × II',64,554,1160,48,30,C.ink,3);next('有了对任务的控制，哪些条件限制了这些变量？');break;}
 case 17:{
 sub('生产者u：QK',64,157,490,1);sub('消费者v：softmax',762,157,450,1);task('写入结果',130,267,380,C.orange,1);task('读取结果',788,267,370,C.green,1);ln(130,346,1180,346,C.line,1);ln(130,334,130,358,C.orange,1);ln(510,332,510,360,C.orange,1);tx('Sᵤ',130,366,180,40,29,C.orange,1);tx('Sᵤ + Lᵤ',448,366,260,40,29,C.orange,1);ln(518,346,780,346,C.green,2,true,3);f('Sᵥ ≥ Sᵤ + Lᵤ',64,454,1150,2,43);f('跨轮依赖：Sᵥ + d × II ≥ Sᵤ + Lᵤ',64,533,1150,3,35);tx('d：消费者与生产者的轮次距离',64,599,1150,38,26,C.muted,3);break;}
 case 18:{
 sub('同一buffer版本：最后一次读取完成后，才能再次写入',64,153,1160,1);task('本轮写入',105,260,210,C.blue,1);task('最后一次读取',448,260,260,C.green,1);task('下一次写入',931,260,260,C.blue,2);ln(105,337,1192,337,C.line,1);tx('S写',105,354,170,42,29,C.blue,1);tx('S末读 + L末读',550,354,330,42,29,C.green,1);tx('S写 + b × II',920,354,300,42,29,C.blue,2);f('S写 + b × II ≥ S末读 + L末读',64,449,1160,2,40);tx('增加buffer份数b，让同一版本更晚被复用',64,525,1160,45,31,C.red,3,true);tx('Σ（buffer份数 × 每份大小）≤ 片上容量',64,590,1160,45,29,C.ink,3);break;}
 case 19:{
 tx('同一硬件资源：占用区间不能重叠',64,151,1160,47,32,C.red,0,true);task('任务i · 占用Hᵢ',194,236,340,C.orange,1);task('任务j · 占用Hⱼ',653,236,340,C.orange,1);ln(194,303,1184,303,C.line,1);tx('rᵢ',194,314,120,42,28,C.orange,1);tx('rⱼ',653,314,120,42,28,C.orange,1);tx('rᵢ + II',1060,314,150,42,28,C.orange,1);sub('如果 i 在 j 前',64,392,540,2);tx('rⱼ − rᵢ ≥ Hᵢ\nrᵢ + II − rⱼ ≥ Hⱼ',64,450,550,126,33,C.ink,2);sub('如果 j 在 i 前',700,392,500,3);tx('rᵢ − rⱼ ≥ Hⱼ\nrⱼ + II − rᵢ ≥ Hᵢ',700,450,510,126,33,C.ink,3);tx('第一式限制本周期；第二式限制跨周期边界',64,604,1150,39,26,C.muted,3);break;}
 case 20:{
 tx('已知：L、H、d、buffer大小与容量',64,150,680,40,25,C.blue,1,true);tx('未知：k、r、b',760,150,280,40,25,C.purple,1,true);tx('目标：min II',1045,150,180,40,25,C.red,1,true);
 const eq=[['时间表示','Sᵢ = kᵢ × II + rᵢ'],['数据依赖','Sᵥ + d × II ≥ Sᵤ + Lᵤ'],['buffer复用','S写 + b × II ≥ S末读 + L末读'],['资源互斥','i在前：rⱼ − rᵢ ≥ Hᵢ；rᵢ + II − rⱼ ≥ Hⱼ'],['取值与容量','k ≥ 0；0 ≤ r < II；b ≥ 1；Σ b × 大小 ≤ 容量']];
 eq.forEach(([a,b],i)=>{tx(a,64,221+i*61,195,45,25,C.muted,1);(()=>{const e=tx(b,270,221+i*61,940,45,i>=3?27:31,C.ink,1);e.text=[b.split(/(II|L末读|S末读|S写|[LHdrkSb][ᵢᵤᵥⱼ]?|大小|容量)/g).filter(Boolean).map(t=>({run:t,style:{color:t==='II'?C.red:/^[LHd]|大小|容量/.test(t)?C.blue:/^[krSb]|写|末读/.test(t)?C.purple:C.ink}}))];return e;})()});
 tx('反向顺序交换i、j；变量为整数',64,525,1160,30,21,C.muted,1);tx('k × II、b × II：两个未知量相乘',64,558,1160,48,34,C.red,2,true);tx('直接优化min II：非线性求解成本高，难以控制编译时间',64,614,1160,42,28,C.ink,3);break;}
 case 21:{
 tx('先给定II，再判断：k、r、b是否存在可行解？',64,153,1160,61,35,C.red,1,true);tx('已知常数 × 未知量 → 线性整数约束',64,223,1160,45,29,C.ink,1);ln(135,338,1140,338,C.ink,2,false,3);tx('下界',89,291,200,38,26,C.muted,2);tx('上界',1040,291,170,38,26,C.muted,2);bx('候选II',520,306,240,62,C.red,2,false,29);ln(640,378,640,426,C.red,2,true,3);bx('Z3：检查约束是否可解',390,438,500,75,C.purple,2,false,31);ln(389,475,220,475,C.green,3,true,3);tx('可行：尝试更小II',64,535,560,50,31,C.green,3,true);ln(895,475,1100,475,C.red,4,true,3);tx('不可行：增大II',747,535,470,50,31,C.red,4,true);next('搜索可行的最小启动间隔，并取出对应调度变量。');break;}
 case 22:{
 sub('串行指令顺序',64,154,1150,1);const a=['QK₀','softmax₀','PV₀','QK₁','softmax₁','PV₁'];a.forEach((t,i)=>bx(t,64+i*193,212,176,52,i%3===1?C.green:C.orange,1,false,24));sub('重排后的Scalar指令顺序',64,310,1160,2);const b=['QK₀','QK₁','softmax₀','PV₀','softmax₁','PV₁'];b.forEach((t,i)=>bx(t,64+i*193,371,176,52,t.includes('soft')?C.green:C.orange,2,false,24));bx('buffer 0：QK₀结果',166,517,415,70,C.blue,3,true,28);bx('buffer 1：QK₁结果',720,517,415,70,C.blue,3,false,28);ln(152,430,290,507,C.blue,3,true,2.5);ln(344,430,856,507,C.blue,3,true,2.5);ln(549,430,430,507,C.green,4,true,2.5);tx('改写buffer索引，并自动插入数据就绪与复用同步',64,615,1160,43,28,C.red,4,true);break;}
 case 23:{
 sub('不排流水',64,155,530,1);sub('排流水',700,155,520,2);for(let side=0;side<2;side++){const x=64+side*636;['搬运','Cube','Vector'].forEach((t,j)=>{tx(t,x,235+j*81,100,39,23,C.muted,side?2:1);ln(x+108,279+j*81,x+(side?512:548),279+j*81,C.line,side?2:1)});if(!side){const labels=['入₀','QK₀','归一₀','PV₀','入₁','QK₁','归一₁','PV₁'];const rows=[0,1,2,1,0,1,2,1];labels.forEach((t,z)=>{const label=bx(t,x+108+z*55,235+rows[z]*81,58,39,[C.blue,C.orange,C.green][rows[z]],1,false,16);if(t==='归一₀'||t==='归一₁'){label.text.insets={top:5,right:2,bottom:5,left:2};label.text.style={typeface:'Microsoft YaHei',fontSize:16,color:C.white,bold:false,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'none'};}});}else{task('入₀',x+108,235,70,C.blue,2);task('入₁',x+182,235,70,C.blue,2);task('QK₀',x+181,316,80,C.orange,2);task('QK₁',x+265,316,80,C.orange,2);task('PV₀',x+349,316,80,C.orange,2);task('PV₁',x+433,316,80,C.orange,2);task('归一₀',x+265,397,80,C.green,2);task('归一₁',x+349,397,80,C.green,2)}}
 tx('Cube MFU',64,510,1150,50,35,C.red,3,true,'center');tx('23.35%  →  90.77%',64,573,1150,78,62,C.red,3,true,'center');break;}
 case 24:tx('谢谢，欢迎提问',100,283,1080,126,70,C.red,0,true,'center');break;
 }
 s.speakerNotes.text=notes[n]||`P${String(n).padStart(2,'0')} ${titles[n-1]}`;
 await fs.writeFile(`${ROOT}/manifest-${String(n).padStart(2,'0')}.json`,JSON.stringify({slide:n,title:titles[n-1],clicks:clicks[n-1],animations,items},null,2));out.push({n,title:titles[n-1],clicks:clicks[n-1]});
 }
 return out;
}
