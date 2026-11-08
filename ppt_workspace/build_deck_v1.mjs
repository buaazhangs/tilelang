import fs from 'node:fs/promises';

// All technical diagrams are native editable slide objects.
// Cxx in each shape name maps to the approved click number.
const ROOT='D:/Dev/tilelang/ppt_workspace';
const C={bg:'#F6F4EF',ink:'#172A3A',muted:'#60717D',line:'#CCD4D8',blue:'#397EB8',blueL:'#E3EEF7',orange:'#D97D29',orangeL:'#F8E9D8',green:'#54836A',greenL:'#E3EEE7',purple:'#7762A8',purpleL:'#EBE6F3',red:'#B84E46',redL:'#F6E5E0',white:'#FFFFFF',gray:'#E4E5E2'};
const TITLES=['让昇腾的机器不闲着','FlashAttention 的五道工序','时间花在哪里','让不同轮次重叠','GPU 的工人与工头','就绪的工作填补空档','昇腾按排班表交接','排班质量决定等待','TileLang 的职责','一份周期性重复的排班','编译器需要决定什么','数据就绪，才能开始','货位复用也有先后','一个周期，两条资源约束','固定启动间隔，简化求解','二分搜索可行的启动间隔','排班结果怎样变成程序','五道工序，展开为十三个任务','理想流水最多能快多少','让编译器承担流水编排','谢谢，欢迎提问','为什么可以二分','推导下界与代码初始界'];
const KEYS=['空闲','工序','产能','重叠','工头','见缝插针','排班','负担','职责','节拍','变量','等料','货位','互斥','非线性','二分','落地','展开','收益','易用性','提问','单调性','初始界'];
const NEXT=['一轮计算经过哪些工序？','这些机器都忙起来了吗？','怎样减少空等？','谁来安排这些重叠？','有工人卡住时会怎样？','昇腾怎样组织工作？','排班定下后，执行时还能改变吗？','TileLang 接过哪些工作？','先说清楚重复的节拍','这张排班表需要哪些变量？','这些变量受什么限制？','旧数据用完，货位才能复用','货位之外，机器还有什么限制？','能否直接求最小启动间隔？','启动间隔应该试多大？','求解结果怎样落地？','放大看看同一个 FlashAttention','把任务耗时代入模型','回到研究者的编程负担','讨论与提问','','',''];
const counts=[1,4,3,3,3,4,4,3,3,4,4,4,4,6,3,5,4,3,4,2,0,0,0];
const section=n=>n<=4?0:n<=8?1:n<=18?2:3;

export async function build({presentation,input}){
 const speech=await fs.readFile('D:/Dev/tilelang/PPT交付/配套演讲稿.md','utf8');
 const factoryBytes=new Uint8Array(await fs.readFile(ROOT+'/factory.png'));
 const factorySource=input.local?{blob:factoryBytes,contentType:'image/png'}:{path:ROOT+'/factory.png'};
 const notes={}; for(const m of speech.matchAll(/^## (P\d{2})[^\n]*\n([\s\S]*?)(?=^## |$(?![\s\S]))/gm))notes[Number(m[1].slice(1))]=m[2];
 const out=[];
 for(let n=input.from;n<=input.to;n++){
  const s=n<=presentation.slides.items.length?presentation.slides.getItem(n-1):presentation.slides.add({width:1280,height:720});
  s.shapes.deleteAll(); for(const im of [...s.images.items])im.delete(); for(const ch of [...s.charts.items])presentation.delete(ch.toSnapshot().aid);
  s.background.fill=C.bg;s.hidden=n>21;
  let seq=0;const anim=[];const items=[];
  function name(k,key=''){return `P${String(n).padStart(2,'0')}_C${String(k).padStart(2,'0')}_${String(++seq).padStart(3,'0')}_${key}`;}
  function record(el,k,key,opts={}){const nm=name(k,key);el.name=nm;items.push({name:nm,k,kind:key});if(k)anim.push({name:nm,click:k,effect:'fade',...opts});return el;}
  function shape(x,y,w,h,fill=C.white,k=0,key='box',line='none',geometry='rect'){
   const el=s.shapes.add({geometry,name:name(k,key),position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:line==='none'?0:1.5}});
   const nm=el.name;items.push({name:nm,k,kind:key});if(k)anim.push({name:nm,click:k,effect:'fade'});return el;
  }
  function text(str,x,y,w,h=45,size=26,color=C.ink,k=0,opt={}){
   const el=shape(x,y,w,h,'none',k,opt.key||'text');el.text=str;el.text.style={typeface:opt.mono?'Consolas':'Microsoft YaHei',fontSize:size,color,bold:!!opt.bold,alignment:opt.align||'left',verticalAlignment:opt.valign||'middle',autoFit:'none',wrap:'square'};el.text.insets=0;return el;
  }
  function box(str,x,y,w,h,color,k=0,opt={}){const el=shape(x,y,w,h,opt.light||color,k,opt.key||'task',opt.border||'none',opt.round?'roundRect':'rect');el.text=str;el.text.style={typeface:'Microsoft YaHei',fontSize:opt.size||25,color:opt.textColor||C.white,bold:!!opt.bold,alignment:'center',verticalAlignment:'middle',autoFit:'none'};el.text.insets=8;return el;}
  function path(points,color,k=0,width=2.5,arrow=true,key='arrow'){
   const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||1,h=Math.max(...ys)-y||1;
   const el=s.shapes.add({geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width},customPaths:[{width:w,height:h,commands:points.map((p,i)=>({[i?'lineTo':'moveTo']:{x:p[0]-x,y:p[1]-y}}))}]});record(el,k,key);
   if(arrow){const a=points.at(-2),b=points.at(-1),ang=Math.atan2(b[1]-a[1],b[0]-a[0])*180/Math.PI+90;const head=shape(b[0]-6,b[1]-7,12,14,color,k,'arrowhead','none','triangle');head.position={...head.position,rotation:ang};}
   return el;
  }
  function rule(x,y,w,color=C.line,k=0){return path([[x,y],[x+w,y]],color,k,1.3,false,'rule');}
  function smallLabel(str,x,y,w,color=C.muted,k=0){return text(str,x,y>=595?590:y,w,34,24,color,k);}
  function formula(str,x,y,w,k,color=C.ink,size=35){return text(str,x,y,w,68,size,color,k,{bold:true});}
  function note(str,k=0){text(str,70,582,1140,38,23,C.muted,k);}
  function lane(labels,x=200,y=225,w=930,step=95,k=0){labels.forEach((v,i)=>{text(v,70,y+i*step,120,45,23,C.muted,k);rule(x,y+i*step+48,w,C.line,k);});}
  function task(label,x,y,w,color,k=0,round=0){return box(label,x,y,w,44,color,k,{size:22,round, key:'task'});}
  function circ(label,x,y,r,color,k=0,sz=24){return box(label,x,y,r,r,color,k,{size:sz,key:'node',round:false});}
  function band(label,x,y,w,color,k=0){box(label,x,y,w,50,color,k,{light:color,textColor:C.white,size:24});}
  if(n!==1&&n!==21){
   text(TITLES[n-1],66,46,1090,66,42,C.ink,0,{bold:true});
   text(KEYS[n-1],1060,120,152,36,27,C.purple,0,{align:'right'});
   const sections=['硬件利用率','GPU 与昇腾','编译器如何排流水','收益与易用性'];
   sections.forEach((a,i)=>{text(a,68+i*260,675,240,22,16,i===section(n)?C.ink:C.muted,0,{bold:i===section(n)});});
   rule(68,665,1144);text(n>21?'B'+(n-21):String(n).padStart(2,'0'),1160,672,55,26,17,C.muted,0,{align:'right'});
   if(NEXT[n-1])text(NEXT[n-1],70,628,1140,34,27,C.muted,0);
  }
  switch(n){
   case 1:{
    text('让昇腾的机器不闲着',70,68,1120,100,61,C.ink,0,{bold:true});
    text('TileLang Ascend AutoSchedule 的模调度设计',74,177,1100,52,28,C.muted);
    const im=s.images.add({...factorySource,fit:'contain',position:{left:70,top:275,width:1140,height:380},alt:'蓝色搬运机、橙色矩阵机、绿色向量机的工厂比喻插画'});im.name='P01_C00_factory';
    text('同一个 FlashAttention，为什么机器会闲着？',74,240,1120,45,29,C.purple,1,{bold:true});
    box('空 等',190,320,115,46,C.gray,1,{textColor:C.muted,size:26});box('空 等',1010,320,115,46,C.gray,1,{textColor:C.muted,size:26});
    text('工厂比喻 · 算法到硬件 · 编译期调度',75,669,1100,28,19,C.muted);break;
   }
   case 2:{
    text('O = softmax(QKᵀ / √d) V',70,153,750,50,37,C.ink,1,{bold:true});
    text('固定 Q 块，依次处理 K / V 块',815,166,400,55,24,C.muted,1);
    const labs=['① 算相关性\nQKᵀ','② 更新归一化\nonline softmax','③ 转交权重\npack_p','④ 加权求和\nPV','⑤ 更新输出\naccumulate'];
    const cols=[C.orange,C.green,C.blue,C.orange,C.green];
    labs.forEach((a,i)=>{box(a,70+i*232,270,205,98,cols[i],2,{size:25});if(i<4)path([[278+i*232,319],[294+i*232,319]],C.muted,2);});
    box('矩阵机（Cube）',75,450,300,62,C.orange,3,{size:27});box('向量机（Vector）',480,450,330,62,C.green,3,{size:27});box('搬运贯穿工序',910,450,300,62,C.blue,3,{size:27});
    text('① 与 ④',75,520,300,35,24,C.orange,3,{align:'center'});text('② 与 ⑤',480,520,330,35,24,C.green,3,{align:'center'});text('宏流程内还会展开搬运',910,520,300,35,21,C.blue,3,{align:'center'});
    path([[360,267],[360,252],[465,252],[465,267]],C.green,4,2,true);smallLabel('归一化状态跨轮',310,214,250,C.green,4);
    path([[1064,370],[1064,406],[1182,406],[1182,370]],C.green,4,2,true);smallLabel('输出 O 跨轮',1024,408,200,C.green,4);
    note('Q 的预装与最终写出在循环外；五段宏流程不等于五个调度任务',4);break;
   }
   case 3:{
    lane(['搬运','矩阵 Cube','向量 Vector'],210,215,700,93,1);
    const series=[{l:0,w:90,c:C.blue,t:'搬入'},{l:1,w:100,c:C.orange,t:'QK'},{l:0,w:42,c:C.blue,t:''},{l:2,w:106,c:C.green,t:'softmax'},{l:0,w:42,c:C.blue,t:''},{l:1,w:100,c:C.orange,t:'PV'},{l:0,w:42,c:C.blue,t:''},{l:2,w:68,c:C.green,t:'累加'}];let xx=210;
    for(const q of series){task(q.t,xx,215+q.l*93,q.w,q.c,1);xx+=q.w+3;}
    box('空 等',320,405,90,44,C.gray,2,{textColor:C.muted});box('空 等',440,308,120,44,C.gray,2,{textColor:C.muted});
    text('23.35%',958,209,250,85,52,C.orange,3,{bold:true});text('矩阵机忙碌时间占比',958,295,260,60,23,C.ink,3);
    shape(958,375,240,20,C.gray,3);shape(958,375,56,20,C.orange,3);
    text('实际 TFLOPS = 运算量 ÷ 时间',75,510,1040,45,28,C.ink,3);text('算力达成率 = 实际算力 ÷ 同精度峰值',75,561,1100,38,25,C.muted,3);
    smallLabel('串行模型示意；忙碌比例不等同于 MFU',76,164,1050,C.muted,1);break;
   }
   case 4:{
    lane(['搬入 K','矩阵计算'],215,235,970,105,0);
    task('本轮 K',215,235,155,C.blue);task('本轮 QK',390,340,315,C.orange);
    const move=task('下一轮 K',930,235,180,C.blue);anim.push({name:move.name,click:1,effect:'motion',dx:-535,dy:0,duration:0.75});
    path([[930,213],[395,213]],C.blue,1,3,true);smallLabel('只提前搬运，后续计算仍保留依赖',405,161,700,C.blue,1);
    box('旧 K 还在读，新 K 放哪里？',72,475,540,70,C.red,2,{light:C.redL,textColor:C.red,size:27});
    box('货位 1\n本轮 K',720,465,205,105,C.blue,3,{light:C.blueL,textColor:C.blue,size:25});box('货位 2\n下一轮 K',955,465,205,105,C.blue,3,{size:25});
    text('多留货位以重叠工作；时间安排变化，计算结果不变',72,571,1110,40,27,C.ink,3);break;
   }
   case 5:{
    const ys=[210,305,400];ys.forEach((y,i)=>{box('warp '+(i+1),75,y,160,60,C.blue,1,{size:26});text('QK  →  softmax  →  PV',270,y,410,60,26,C.ink,1);});
    box('工头\nwarp scheduler',755,280,215,115,C.purple,2,{size:28});path([[680,337],[743,337]],C.purple,2);box('共享机器',1020,288,190,100,C.orange,2,{size:26});path([[972,337],[1008,337]],C.orange,2);
    text('工人是指令流，不是执行机器',75,492,650,42,30,C.ink,1,{bold:true});
    ['材料已到','机器能接活','未被同步阻塞'].forEach((t,i)=>{box(t,75+i*382,557,350,54,C.green,3,{light:C.greenL,textColor:C.green,size:25});});break;
   }
   case 6:{
    lane(['warp 1','warp 2','warp 3','矩阵机'],210,185,950,80,0);
    box('等材料',210,185,240,44,C.red,1,{light:C.redL,textColor:C.red});
    task('QK₂',450,265,170,C.orange,2);task('QK₃',640,345,170,C.orange,2);task('QK₂',450,425,170,C.orange,2);task('QK₃',640,425,170,C.orange,2);
    path([[535,310],[535,417]],C.orange,2);path([[725,390],[725,417]],C.orange,2);
    text('谁就绪，就放谁',825,235,360,75,39,C.purple,3,{bold:true});text('前提：还有其他就绪的工作',825,326,370,70,25,C.muted,3);
    text('Hopper 也按任务分工、提前排流水\n寄存器压力限制驻留工人；warp scheduler 仍动态补位',75,533,1120,78,27,C.ink,4);break;
   }
   case 7:{
    box('调度员\nScalar',75,260,190,120,C.purple,1,{size:30});smallLabel('编译期排好的任务顺序',75,202,420,C.purple,1);
    const rows=[{y:210,l:'搬运',c:C.blue,t:['K₀','V₀','K₁']},{y:330,l:'Cube',c:C.orange,t:['QK₀','PV₀','QK₁']},{y:450,l:'Vector',c:C.green,t:['softmax₀','累加₀','softmax₁']}];
    rows.forEach(r=>{path([[280,320],[333,320],[333,r.y+28],[382,r.y+28]],C.muted,2);text(r.l,400,r.y-43,200,35,23,r.c,2);r.t.forEach((v,i)=>box(v,400+i*210,r.y,180,58,r.c,2,{size:25}));path([[1028,r.y+29],[1120,r.y+29]],r.c,2);});
    path([[487,270],[487,312]],C.green,3,3,true);box('完成信号',1065,290,145,64,C.green,3,{size:24});
    text('GPU：运行时挑就绪工单      昇腾：提前编排各资源的顺序',75,555,1140,52,27,C.ink,4);break;
   }
   case 8:{
    smallLabel('矩阵机的队列',75,167,450,C.orange);
    box('PV₀\n仍等 softmax',75,235,315,110,C.red,1,{light:C.redL,textColor:C.red,size:29});box('QK₁\n数据已就绪',440,235,315,110,C.orange,1,{size:29});path([[398,290],[428,290]],C.muted,1);
    text('队头等待，后面的就绪工作也被挡住',810,239,395,110,31,C.red,1,{bold:true});
    text('重新编译时调整排班',75,386,650,40,25,C.purple,2,{bold:true});box('先发 QK₁',75,440,315,62,C.orange,2,{size:28});box('再发 PV₀',440,440,315,62,C.orange,2,{light:C.orangeL,textColor:C.orange,size:28});path([[398,471],[428,471]],C.orange,2);
    text('硬件不会在运行时自动交换顺序',810,438,395,70,25,C.muted,2);
    ['排任务顺序','定货位份数','安排同步交接'].forEach((a,i)=>text(a,75+i*380,558,350,45,29,C.ink,3,{bold:true}));break;
   }
   case 9:{
    text('用户写计算',75,163,520,50,32,C.ink,1,{bold:true});text('编译器排流水',735,163,470,50,32,C.purple,2,{bold:true});
    shape(75,231,575,290,C.white,1);text('for kv in T.Pipelined(...):\n    qk(...)\n    softmax(...)\n    pack_p(...)\n    pv(...)\n    accumulate_output(...)',100,247,535,250,25,C.ink,1,{mono:true});
    ['分析依赖','估计任务代价','决定阶段、顺序、份数','自动插入同步'].forEach((a,i)=>{text(a,745,235+i*68,460,48,29,i===2?C.purple:C.ink,2,{bold:i===2});if(i<3)path([[950,283+i*68],[950,298+i*68]],C.purple,2);});
    text('手工：填写阶段与份数       自动：交给求解器',75,553,1120,48,30,C.purple,3,{bold:true});
    smallLabel('依据 core.py 简化；基本任务可自动识别，需要时用 T.Task 分组',75,601,1120,C.muted,1);break;
   }
   case 10:{
    const x=210,y=215,dx=160;['第0轮','第1轮','第2轮','第3轮'].forEach((a,i)=>{text(a,75,y+i*62,120,42,23,C.muted,1);task('搬运',x+i*dx,y+i*62,85,C.blue,1);task('QK',x+90+i*dx,y+i*62,85,C.orange,1);task('归一化',x+180+i*dx,y+i*62,100,C.green,1);});
    path([[210,192],[370,192]],C.purple,2,3);text('II',260,149,70,36,30,C.purple,2,{bold:true});
    text('启动',210,484,170,38,25,C.muted,3);text('稳态：重复同一份排班',410,484,500,38,25,C.ink,3);text('收尾',1010,484,140,38,25,C.muted,3);
    text('模调度：为循环寻找周期性重复的排班',75,531,1135,40,26,C.purple,3,{bold:true});
    formula('总时间 ≈ 首轮时长 + (N − 1) × II',75,574,1140,4,C.ink,30);break;
   }
   case 11:{
    const x=75,w=285;for(let i=0;i<4;i++){shape(x+i*w,245,w-8,105,i===2?C.purpleL:C.white,1,'window',C.line);text('k = '+i,x+12+i*w,207,w-24,34,25,C.muted,1);}
    const block=task('任务 i',x+2*w+80,268,150,C.purple,1);path([[x+2*w,372],[x+2*w+80,372]],C.purple,1);text('r',x+2*w+28,382,40,35,28,C.purple,1);
    formula('Sᵢ = kᵢ × II + rᵢ',75,434,760,2,C.ink,42);text('0 ≤ rᵢ < II',860,448,340,45,30,C.muted,2);
    text('同一资源：按周期内位置 r 排序',75,520,1130,45,29,C.orange,3);
    ['阶段 k：跨几个节拍','位置 r：周期内先后','份数 b：存储几份数据'].forEach((a,i)=>box(a,75+380*i,578,355,42,C.purple,4,{light:C.purpleL,textColor:C.purple,size:27,bold:true}));break;
   }
   case 12:{
    box('生产者 u\nQK 结果就绪',75,200,320,100,C.orange,1,{size:29});box('消费者 v\n在线 softmax',835,200,365,100,C.green,1,{size:29});path([[410,250],[815,250]],C.muted,1);
    text('等待时长',485,201,280,40,27,C.muted,1,{align:'center'});rule(75,360,1120);path([[75,349],[75,371]],C.orange,1,3,false);path([[620,344],[620,371]],C.green,1,3,false);path([[620,360],[1185,360]],C.green,1,3,true);text('Sᵤ',75,372,150,42,30,C.orange,1);text('最早可读边界',580,372,300,42,26,C.green,1);
    formula('同一轮： Sᵥ ≥ Sᵤ + 等待时长',75,433,1140,2,C.ink,36);
    formula('跨轮： Sᵥ + d × II ≥ Sᵤ + 等待时长',75,514,1140,3,C.ink,33);
    text('d = 1：上一轮更新的归一化状态，供下一轮读取',75,587,1140,40,25,C.green,3);
    text('结果准备好，后续访问才安全',75,158,1120,37,25,C.muted,4);break;
   }
   case 13:{
    const labels=['第0轮','第1轮','第2轮'];labels.forEach((a,i)=>{text(a,75+i*380,168,350,45,25,C.muted,1);box(i===2?'回到货位 0':'货位 '+i,75+i*380,230,330,100,i===2?C.red:C.blue,1,{light:i===2?C.redL:C.blueL,textColor:i===2?C.red:C.blue,size:29});});
    path([[240,340],[240,371],[1060,371],[1060,340]],C.red,1,2,true);text('旧数据读完，才能再次写入同一格',300,375,760,45,28,C.red,1,{align:'center'});
    formula('S写 + b × II ≥ S读 + 等待时长',75,446,1135,2,C.ink,36);
    shape(224,331,880,94,C.bg,3,'clear_old_reuse');box('货位 2',835,230,330,100,C.blue,3,{light:C.blueL,textColor:C.blue,size:29});text('第 2 轮使用新货位；货位 0 到第 3 轮才复用',300,375,850,45,26,C.blue,3,{align:'center'});
    text('增加到 3 份：同一货位再晚一轮复用',75,521,1140,43,28,C.blue,3,{bold:true});
    text('容量约束：Σ（份数 × 每份大小）≤ 片上容量',75,583,1140,45,28,C.purple,4);break;
   }
   case 14:{
    // Native cyclic schedule diagram: two resource arcs and a wrap arrow.
    const cx=300,cy=350,r=140;
    function arc(a,b,col,k){let pts=[];for(let q=a;q<=b;q+=3){const th=q*Math.PI/180;pts.push([cx+r*Math.cos(th),cy+r*Math.sin(th)]);}path(pts,col,k,22,false,'resource_arc');}
    arc(-90,70,C.orange,1);arc(100,250,C.purple,2);
    text('QK',327,287,100,50,31,C.orange,1,{bold:true});text('PV',190,375,100,50,31,C.purple,2,{bold:true});text('周期 II',215,340,190,40,27,C.ink,1,{align:'center'});
    path([[243,194],[243,168],[300,168],[300,196]],C.muted,3,2,true);text('跨周期',82,167,145,38,27,C.muted,3);text('同一排班每隔 II 重复',140,525,370,38,27,C.muted,3);
    formula('rPV − rQK ≥ qQK',560,201,650,2,C.orange,36);text('本周期内，不占用同一段资源',560,265,650,38,27,C.muted,2);
    formula('rQK + II − rPV ≥ qPV',560,331,650,3,C.purple,35);text('末尾任务，也要让出下一周期的开头',560,395,650,38,27,C.muted,3);
    text('若 PV 在前，交换两者；求解器选择顺序',560,451,650,42,25,C.muted,4);
    formula('II ≥ qQK + qPV',560,505,650,5,C.ink,37);
    text('q：机器接收下一项工作前的占用间隔；与结果就绪等待不同',75,590,1140,36,23,C.muted,6);break;
   }
   case 15:{
    text('直接优化 II',75,175,500,50,31,C.ink,1,{bold:true});formula('k × II',75,250,440,1,C.purple,54);formula('b × II',75,350,440,1,C.purple,54);text('两个未知量相乘',75,460,450,45,29,C.red,1);
    path([[505,337],[665,337]],C.muted,2,4,true);text('通常更难求解',500,267,225,48,26,C.muted,2,{align:'center'});
    text('先固定 II = 1200（示意）',750,175,480,50,31,C.ink,3,{bold:true});formula('1200 × k',750,250,470,3,C.purple,47);formula('1200 × b',750,350,470,3,C.purple,47);
    text('常数 × 未知量',750,460,470,45,29,C.green,3);text('剩下的仍是整数变量与先后选择，并非普通连续线性规划',75,565,1130,50,26,C.muted,3);break;
   }
   case 16:{
    const xx=100,ww=1080;rule(xx,260,ww,C.ink,1);text('资源工作量下界',75,193,410,43,28,C.muted,1);text('串行上界',900,193,300,43,28,C.muted,1,{align:'right'});
    box('候选 II',512,230,240,60,C.purple,2,{size:28});path([[632,303],[632,355]],C.purple,2,3,true);box('Z3 可行性检查',390,375,485,90,C.purple,2,{size:33});
    text('能排：尝试更小的 II',75,490,570,52,29,C.green,2,{bold:true});text('不能排：放宽 II',710,490,500,52,29,C.red,3,{bold:true});
    path([[420,305],[215,305]],C.green,2,4,true);path([[810,305],[1010,305]],C.red,3,4,true);
    text('每次缩小一半范围，寻找最小可行 II',75,557,1140,45,29,C.ink,4,{bold:true});
    smallLabel('基础模型内：可行节拍放宽后仍可行；证明与工程边界见备用页',75,602,1140,C.muted,5);break;
   }
   case 17:{
    text('求解结果',75,169,280,48,32,C.purple,1,{bold:true});box('每个任务：k、r\n每个缓冲区：b',75,242,290,165,C.purple,1,{light:C.purpleL,textColor:C.purple,size:28});
    path([[375,323],[445,323]],C.purple,2,3,true);text('程序结构与顺序',480,169,460,48,32,C.ink,2,{bold:true});
    ['启动：先启动早期阶段','稳态：交错不同轮次','收尾：完成剩余工作'].forEach((a,i)=>box(a,480,237+i*76,445,58,C.ink,2,{light:i===1?C.purpleL:C.white,textColor:C.ink,size:26}));
    text('r 决定周期内的发射顺序',480,492,650,45,27,C.purple,2);
    box('轮换货位\nn mod b',985,240,225,140,C.blue,3,{light:C.blueL,textColor:C.blue,size:29});text('自动插入同步',985,435,225,75,27,C.green,4,{bold:true});
    text('模型时间用于规划；实际开工由就绪与同步控制',75,564,1140,45,27,C.muted,4);break;
   }
   case 18:{
    const rows=[['搬入',C.blue,['T0  K  518','T7  V  518']],['片内送料',C.blue,['T1  133','T2  133','T8  133','T9  133']],['矩阵',C.orange,['T3  QK  520','T10  PV  520']],['结果搬出',C.blue,['T4  311','T11  311']],['向量',C.green,['T5  softmax  735','T12  累加  318']],['权重回传',C.blue,['T6  171']]];
    text('五段宏流程：QK / softmax / 转交权重 / PV / 累加',75,157,1130,45,28,C.ink,1);
    rows.forEach((r,j)=>{const y=222+j*60;text(r[0],75,y,150,44,27,r[1],2);r[2].forEach((a,i)=>{const label=a.split(/\s+/).slice(0,-1).join(' ');shape(250+i*238,y,213,44,r[1],2,'task');text(label,260+i*238,y,145,44,22,C.white,2);text(a.split(/\s+/).at(-1),405+i*238,y,48,44,24,C.white,3,{align:'right'});});});
    smallLabel('周期数来自参考材料的耗时估计；735 / 318 为源码中的显式提示',75,595,1140,C.muted,3);break;
   }
   case 19:{
    const vals=[1036,532,1040,622,1053,171],labs=['搬入','片内送料','矩阵','结果搬出','向量','权重回传'];
    const ch=s.charts.add('bar',{position:{left:75,top:178,width:665,height:344},categories:labs,series:[{name:'每轮资源工作量',values:vals,fill:C.blue}],hasLegend:false,barOptions:{direction:'bar',grouping:'clustered',gapWidth:65},xAxis:{textStyle:{fontSize:20,typeface:'Microsoft YaHei'}},yAxis:{min:0,max:1200,majorUnit:400,majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:20}},chartFill:C.bg,plotAreaFill:C.bg});record(ch,1,'chart');
    text('II ≥ 1053',790,192,400,67,47,C.green,1,{bold:true});text('资源下界，由 Vector 决定',790,264,415,40,24,C.muted,1);
    text('4454 ÷ 1053',790,339,400,52,34,C.ink,2);text('≤ 4.23 倍',790,398,420,78,55,C.purple,2,{bold:true});text('长循环理想加速上限',790,478,410,38,25,C.purple,2);
    text('Cube 理论忙碌比例：23.35% → 98.77%',75,545,1135,48,30,C.orange,3,{bold:true});
    text('首尾开销 · 有限货位 · 依赖与估计误差',75,597,1135,32,24,C.muted,4);
    smallLabel('假设：占用间隔取参考耗时，忽略其他依赖与容量限制；非实测',75,145,1130,C.muted,1);break;
   }
   case 20:{
    const ys=[205,320,435],cs=[C.purple,C.blue,C.green],a=['排顺序','定货位','管交接'],b=['阶段 k、位置 r','缓冲区份数 b','自动同步'];
    a.forEach((v,i)=>{text(v,75,ys[i],320,65,42,cs[i],1,{bold:true});path([[410,ys[i]+34],[565,ys[i]+34]],cs[i],1,3,true);text(b[i],620,ys[i],580,65,37,C.ink,1);});
    text('默认自动，需要时用 T.Stage、buffer 份数等提示干预',75,570,1130,45,28,C.muted,2);break;
   }
   case 21:{
    text('谢谢，欢迎提问',75,78,1120,100,64,C.ink,0,{bold:true});text('算法表达清楚，让编译器安排流水',80,192,1120,60,31,C.purple);
    s.images.add({...factorySource,fit:'contain',position:{left:75,top:315,width:1120,height:330},alt:'与开场相同的三种机器工厂'});
    text('硬件为什么等待      谁来安排工作      编译器怎样求解',80,268,1120,40,26,C.muted);break;
   }
   case 22:{
    text('基础模型：等待时长 ≥ 0，0 ≤ r < II',75,169,1130,55,31,C.muted);
    formula('c × II + (rᵥ − rᵤ) ≥ 等待时长',75,252,1130,0,C.ink,38);text('其中 c = kᵥ − kᵤ + d，是整数',75,326,1130,46,28,C.muted);
    text('若 c ≤ −1，由 rᵥ − rᵤ < II，左侧会小于 0，矛盾',75,405,1130,55,30,C.red);text('因此 c ≥ 0；增大 II 不会破坏这条数据约束',75,481,1130,50,30,C.green,0,{bold:true});
    text('资源绕回约束变松，容量约束不变；此证明不覆盖所有工程配置',75,562,1130,50,25,C.muted);break;
   }
   case 23:{
    text('讲解中推导的下界',75,173,1100,45,30,C.ink,0,{bold:true});formula('II ≥ 最忙资源每轮的工作量',75,231,1130,0,C.purple,39);
    text('代码的初始搜索范围',75,339,1100,45,30,C.ink,0,{bold:true});text('下界：max(iis)\n上界：max(sum(latencies), sum(iis)) + 1',75,403,1130,105,29,C.ink,0,{mono:true});
    text('固定 II 后做可行性检查；没有继续最小化单轮完成时间',75,550,1130,45,27,C.muted);smallLabel('资源限制可能导致 unknown；自动路径无可行模型时有串行回退',75,600,1130,C.muted);break;
   }
  }
  s.speakerNotes.text=(notes[n]||`备用页：${TITLES[n-1]}`)+'\n\n来源：D:/Dev/tilelang/PPT设计框图.md；演讲稿.md；worker.md；autoschedule_time_constraints.md；参考资料2.txt；tilelang/ascend/transform/z3_scheduler.py；examples/ascend/flash_attention/core.py。\nGPU机制： https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html\n工厂插画：AI生成的教学比喻图，不是真实硬件照片。\n数值为参考耗时下的模型推演，非实测。';
  const manifest={slide:n,title:TITLES[n-1],clicks:counts[n-1],animations:anim,items};
  await fs.writeFile(`${ROOT}/manifest-${String(n).padStart(2,'0')}.json`,JSON.stringify(manifest,null,2));out.push({slide:n,elements:items.length,clicks:counts[n-1]});
 }
 return out;
}
