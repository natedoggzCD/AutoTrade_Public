[CmdletBinding()]
param(
    [string]$ConfigPath = ".mcp.json",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Invoke-CodexCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [switch]$AllowFailure
    )

    if ($DryRun) {
        Write-Host ("DRY RUN: codex " + ($Arguments -join " "))
        return 0
    }

    & codex @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $AllowFailure) {
        throw "codex command failed (exit $exitCode): codex $($Arguments -join ' ')"
    }
    return $exitCode
}

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "codex CLI was not found in PATH."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedConfigPath = if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $repoRoot $ConfigPath
}

if (-not (Test-Path $resolvedConfigPath)) {
    throw "MCP config file not found: $resolvedConfigPath"
}

$config = Get-Content -Raw $resolvedConfigPath | ConvertFrom-Json
if (-not $config.mcpServers) {
    throw "No 'mcpServers' object found in $resolvedConfigPath"
}

$servers = @($config.mcpServers.PSObject.Properties)
if ($servers.Count -eq 0) {
    Write-Host "No MCP servers found in $resolvedConfigPath"
    exit 0
}

foreach ($serverProp in $servers) {
    $name = $serverProp.Name
    $server = $serverProp.Value

    Write-Host "Syncing MCP server '$name'..."

    Invoke-CodexCommand -Arguments @("mcp", "remove", $name) -AllowFailure | Out-Null

    $addArgs = @("mcp", "add", $name)

    if ($server.PSObject.Properties.Name -contains "url" -and $server.url) {
        $addArgs += "--url"
        $addArgs += [string]$server.url
        if ($server.PSObject.Properties.Name -contains "bearerTokenEnvVar" -and $server.bearerTokenEnvVar) {
            $addArgs += "--bearer-token-env-var"
            $addArgs += [string]$server.bearerTokenEnvVar
        }
    } else {
        if (-not $server.command) {
            throw "Server '$name' is missing required 'command'."
        }

        if ($server.PSObject.Properties.Name -contains "env" -and $server.env) {
            foreach ($envVar in $server.env.PSObject.Properties) {
                $addArgs += "--env"
                $addArgs += "$($envVar.Name)=$($envVar.Value)"
            }
        }

        $addArgs += "--"
        $addArgs += [string]$server.command

        if ($server.PSObject.Properties.Name -contains "args" -and $server.args) {
            foreach ($arg in $server.args) {
                $addArgs += [string]$arg
            }
        }
    }

    Invoke-CodexCommand -Arguments $addArgs | Out-Null
}

Write-Host "Synced $($servers.Count) MCP servers from $resolvedConfigPath"
if ($DryRun) {
    Write-Host "Run without -DryRun to apply, then run: codex mcp list"
} else {
    Invoke-CodexCommand -Arguments @("mcp", "list") | Out-Null
}
