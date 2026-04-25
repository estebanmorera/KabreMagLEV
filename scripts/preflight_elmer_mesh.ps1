param(
  [string]$GeoPath = "simulations/elmer/base/StepsHTX.geo",
  [string]$OutDir = "",
  [string]$GmshPath = "C:/Program Files/gmsh/gmsh.exe",
  [string]$ElmerGridPath = "C:/Program Files/Elmer 9.0-Release/bin/ElmerGrid.exe"
)

$ErrorActionPreference = "Stop"

function Resolve-FromRoot {
  param([string]$PathValue)

  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return (Resolve-Path -LiteralPath $PathValue).Path
  }

  $repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
  return (Resolve-Path -LiteralPath (Join-Path $repoRoot $PathValue)).Path
}

function Add-Count {
  param(
    [hashtable]$Map,
    [string]$Key,
    [int]$Delta = 1
  )

  if ($Map.ContainsKey($Key)) {
    $Map[$Key] += $Delta
  } else {
    $Map[$Key] = $Delta
  }
}

function Wait-ForFileStable {
  param(
    [string]$PathValue,
    [int]$TimeoutSeconds = 120
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  $lastSize = -1
  $stableChecks = 0

  while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $PathValue) {
      $size = (Get-Item -LiteralPath $PathValue).Length
      if ($size -gt 0 -and $size -eq $lastSize) {
        $stableChecks += 1
        if ($stableChecks -ge 2) { return }
      } else {
        $stableChecks = 0
        $lastSize = $size
      }
    }
    Start-Sleep -Seconds 1
  }

  throw "Timed out waiting for file to be written: $PathValue"
}

$geo = Resolve-FromRoot $GeoPath
$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutDir = Join-Path $repoRoot "simulations/elmer/generated/preflight_$stamp"
} elseif (-not [System.IO.Path]::IsPathRooted($OutDir)) {
  $OutDir = Join-Path $repoRoot $OutDir
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$mshPath = Join-Path $OutDir "mesh.msh"

Write-Host "Generating mesh from $geo"
& $GmshPath $geo -3 -format msh2 -o $mshPath -v 2
Wait-ForFileStable $mshPath

Push-Location $OutDir
try {
  Write-Host "Converting mesh with ElmerGrid"
  & $ElmerGridPath 14 2 "mesh.msh" -out "mesh"
} finally {
  Pop-Location
}

$meshDir = Join-Path $OutDir "mesh"
$elementsPath = Join-Path $meshDir "mesh.elements"
$boundaryPath = Join-Path $meshDir "mesh.boundary"
$nodesPath = Join-Path $meshDir "mesh.nodes"

$elementBody = @{}
$bodyCounts = @{}
Get-Content -LiteralPath $elementsPath | ForEach-Object {
  $parts = $_.Trim() -split "\s+"
  if ($parts.Length -lt 3) { return }
  $elementId = [int]$parts[0]
  $bodyId = [int]$parts[1]
  $elementBody[$elementId] = $bodyId
  Add-Count $bodyCounts ([string]$bodyId)
}

$nodes = @{}
Get-Content -LiteralPath $nodesPath | ForEach-Object {
  $parts = $_.Trim() -split "\s+"
  if ($parts.Length -lt 5) { return }
  $nodes[[int]$parts[0]] = [double[]]@([double]$parts[2], [double]$parts[3], [double]$parts[4])
}

$boundaries = @{}
Get-Content -LiteralPath $boundaryPath | ForEach-Object {
  $parts = $_.Trim() -split "\s+"
  if ($parts.Length -lt 6) { return }

  $tag = [int]$parts[1]
  if (-not $boundaries.ContainsKey($tag)) {
    $boundaries[$tag] = [ordered]@{
      Boundary = $tag
      Elements = 0
      PairCounts = @{}
      TypeCounts = @{}
      XMin = [double]::PositiveInfinity
      XMax = [double]::NegativeInfinity
      YMin = [double]::PositiveInfinity
      YMax = [double]::NegativeInfinity
      ZMin = [double]::PositiveInfinity
      ZMax = [double]::NegativeInfinity
    }
  }

  $record = $boundaries[$tag]
  $record.Elements += 1
  Add-Count $record.TypeCounts ([string]$parts[4])

  $parentA = $elementBody[[int]$parts[2]]
  $parentB = 0
  if ([int]$parts[3] -gt 0) {
    $parentB = $elementBody[[int]$parts[3]]
  }

  if ($parentB -gt 0) {
    $pair = (@($parentA, $parentB) | Sort-Object) -join "-"
  } else {
    $pair = "$parentA-exterior"
  }
  Add-Count $record.PairCounts $pair

  for ($i = 5; $i -lt $parts.Length; $i++) {
    $coord = $nodes[[int]$parts[$i]]
    if ($null -eq $coord) { continue }
    $record.XMin = [Math]::Min($record.XMin, $coord[0])
    $record.XMax = [Math]::Max($record.XMax, $coord[0])
    $record.YMin = [Math]::Min($record.YMin, $coord[1])
    $record.YMax = [Math]::Max($record.YMax, $coord[1])
    $record.ZMin = [Math]::Min($record.ZMin, $coord[2])
    $record.ZMax = [Math]::Max($record.ZMax, $coord[2])
  }
}

$bodyRows = $bodyCounts.GetEnumerator() |
  Sort-Object { [int]$_.Key } |
  ForEach-Object {
    [PSCustomObject]@{
      Body = [int]$_.Key
      Elements = $_.Value
    }
  }

$boundaryRows = $boundaries.Keys |
  Sort-Object |
  ForEach-Object {
    $record = $boundaries[$_]
    $pairs = ($record.PairCounts.GetEnumerator() |
      Sort-Object Name |
      ForEach-Object { "$($_.Name):$($_.Value)" }) -join "; "
    $types = ($record.TypeCounts.GetEnumerator() |
      Sort-Object Name |
      ForEach-Object { "$($_.Name):$($_.Value)" }) -join "; "

    [PSCustomObject]@{
      Boundary = $record.Boundary
      Elements = $record.Elements
      ElementTypes = $types
      ParentPairs = $pairs
      XMin = $record.XMin
      XMax = $record.XMax
      YMin = $record.YMin
      YMax = $record.YMax
      ZMin = $record.ZMin
      ZMax = $record.ZMax
    }
  }

$bodyCsv = Join-Path $OutDir "body_counts.csv"
$boundaryCsv = Join-Path $OutDir "boundary_report.csv"
$bodyRows | Export-Csv -NoTypeInformation -Path $bodyCsv
$boundaryRows | Export-Csv -NoTypeInformation -Path $boundaryCsv

Write-Host ""
Write-Host "Body counts:"
$bodyRows | Format-Table -AutoSize

Write-Host "Likely exterior air boundaries:"
$boundaryRows |
  Where-Object { $_.ParentPairs -match "30-exterior" -and $_.ElementTypes -match "(^|; )303:" } |
  Format-Table Boundary, Elements, ElementTypes, ParentPairs, XMin, XMax, YMin, YMax, ZMin, ZMax -AutoSize

Write-Host "Wrote:"
Write-Host "  $bodyCsv"
Write-Host "  $boundaryCsv"
