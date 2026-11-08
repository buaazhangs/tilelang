param([string]$InputPath='D:\Dev\tilelang\ppt_workspace\static-embedded.pptx', [string]$OutputPath='D:\Dev\tilelang\ppt_workspace\animated-draft.pptx')
$ErrorActionPreference='Stop'
$root='D:\Dev\tilelang\ppt_workspace'
$pp=New-Object -ComObject PowerPoint.Application
try {
  $deck=$pp.Presentations.Open($InputPath,0,0,0)
  $report=@()
  for($n=1;$n -le $deck.Slides.Count;$n++){
    $slide=$deck.Slides.Item($n)
    $manifest=Get-Content -LiteralPath (Join-Path $root ('manifest-{0:00}.json' -f $n)) -Raw | ConvertFrom-Json

    while($slide.TimeLine.MainSequence.Count -gt 0){$slide.TimeLine.MainSequence.Item(1).Delete()}
    $shapeMap=@{};foreach($sh in $slide.Shapes){$shapeMap[$sh.Name]=$sh}
    if($n -eq 19){
      foreach($sh in $slide.Shapes){if($sh.HasChart -eq -1){
        $sh.Chart.ChartArea.Font.Name='Microsoft YaHei'
        $sh.Chart.ChartArea.Font.Size=16
        $sh.Chart.SeriesCollection(1).DataLabels().Font.Name='Microsoft YaHei'
        $sh.Chart.SeriesCollection(1).DataLabels().Font.Size=18
      }}
    }
    $lastClick=0
    foreach($a in ($manifest.animations | Sort-Object click)){
      $sh=$shapeMap[$a.name]
      if($null -eq $sh){throw "Missing shape $($a.name)"}
      $trigger=2;if($a.click -ne $lastClick){$trigger=1;$lastClick=$a.click}
      if($a.effect -eq 'motion'){
        $effect=$slide.TimeLine.MainSequence.AddEffect($sh,0,0,$trigger)
        $behavior=$effect.Behaviors.Add(1)
        $dx=([double]$a.dx/1280).ToString('0.########',[System.Globalization.CultureInfo]::InvariantCulture)
        $dy=([double]$a.dy/720).ToString('0.########',[System.Globalization.CultureInfo]::InvariantCulture)
        $behavior.MotionEffect.Path="M 0 0 L $dx $dy E"
        $effect.Timing.Duration=[single]$a.duration
      } else {
        $effect=$slide.TimeLine.MainSequence.AddEffect($sh,10,0,$trigger)
        $effect.Timing.Duration=0.28
        if($a.effect -eq 'exit'){$effect.Exit=-1}
      }
      $effect.Timing.TriggerDelayTime=0
    }
    $clicks=0;for($j=1;$j -le $slide.TimeLine.MainSequence.Count;$j++){if($slide.TimeLine.MainSequence.Item($j).Timing.TriggerType -eq 1){$clicks++}}
    if($clicks -ne $manifest.clicks){throw "Slide $n click count $clicks != $($manifest.clicks)"}
    $slide.SlideShowTransition.AdvanceOnTime=0
    $slide.SlideShowTransition.AdvanceOnClick=-1
    if($n -gt 21){$slide.SlideShowTransition.Hidden=-1}
    $report += [pscustomobject]@{slide=$n;shapes=$slide.Shapes.Count;effects=$slide.TimeLine.MainSequence.Count;clicks=$clicks;notes=$slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text.Length}
  }
  $deck.SaveAs($OutputPath,24)
  $renderDir=Join-Path $root 'powerpoint-renders'
  New-Item -ItemType Directory -Path $renderDir -Force | Out-Null
  for($n=1;$n -le $deck.Slides.Count;$n++){$deck.Slides.Item($n).Export((Join-Path $renderDir ('slide-{0:00}.png' -f $n)),'PNG',1600,900)}
  $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'animation-audit.json') -Encoding UTF8
  $deck.Close()
  $report | Format-Table
} finally {
  if($null -ne $deck){try{$deck.Saved=-1}catch{}}
  $pp.Quit()
  [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($pp)
}
