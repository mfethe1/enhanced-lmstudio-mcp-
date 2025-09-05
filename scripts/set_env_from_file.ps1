Param(
  [string]$EnvFile = ".secrets/.env.local"
)

function Parse-Line([string]$line){
  $s = $line.Trim()
  if([string]::IsNullOrWhiteSpace($s)){ return $null }
  if($s.StartsWith('#')){ return $null }
  if($s.StartsWith('export ')){ $s = $s.Substring(7).Trim() }
  $idx = $s.IndexOf('=')
  if($idx -lt 0){ return $null }
  $k = $s.Substring(0, $idx).Trim()
  $v = $s.Substring($idx+1).Trim()
  if($v.Length -ge 2 -and (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'")))){
    $v = $v.Substring(1, $v.Length-2)
  }
  return @{ Key=$k; Value=$v }
}

function Mask([string]$val){
  if([string]::IsNullOrEmpty($val)){ return "<empty>" }
  $len = $val.Length
  if($len -le 8){ return "****" }
  return $val.Substring(0,4) + "..." + $val.Substring($len-4)
}

$keysOfInterest = @(
  'ANTHROPIC_API_KEY','ANTHROPIC_MODEL','ANTHROPIC_BASE_URL','ANTHROPIC_VERSION',
  'OPENAI_API_KEY','OPENAI_MODEL','OPENAI_BASE_URL',
  'LMSTUDIO_API_KEY','LMSTUDIO_API_BASE','LM_STUDIO_URL','LMSTUDIO_MODEL','MODEL_NAME'
)

if(!(Test-Path -LiteralPath $EnvFile)){
  Write-Host "Env file not found: $EnvFile" -ForegroundColor Yellow
  exit 1
}

$pairs = @{}
Get-Content -LiteralPath $EnvFile | ForEach-Object {
  $kv = Parse-Line $_
  if($null -ne $kv){ $pairs[$kv.Key] = $kv.Value }
}

Write-Host "== Before ==" -ForegroundColor Cyan
foreach($k in $keysOfInterest){
  $cur = [Environment]::GetEnvironmentVariable($k, 'Process')
  if(-not $cur){ $cur = [Environment]::GetEnvironmentVariable($k, 'User') }
  Write-Host ("{0} = {1}" -f $k, (Mask $cur))
}

$changed = @()
foreach($k in $keysOfInterest){
  if($pairs.ContainsKey($k)){
    $v = $pairs[$k]
    # Set for current process and persist for user profile
    [Environment]::SetEnvironmentVariable($k, $v, 'Process')
    [Environment]::SetEnvironmentVariable($k, $v, 'User')
    $changed += $k
  }
}

Write-Host "== After ==" -ForegroundColor Cyan
foreach($k in $keysOfInterest){
  $cur = [Environment]::GetEnvironmentVariable($k, 'Process')
  if(-not $cur){ $cur = [Environment]::GetEnvironmentVariable($k, 'User') }
  Write-Host ("{0} = {1}" -f $k, (Mask $cur))
}

if($changed.Count -gt 0){
  Write-Host ("Updated keys: {0}" -f ($changed -join ', ')) -ForegroundColor Green
}else{
  Write-Host "No keys updated (already matched)." -ForegroundColor Green
}

