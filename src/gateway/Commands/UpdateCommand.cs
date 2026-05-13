using System.Diagnostics;
using System.Net.Http.Json;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Semantics.Gateway.Infrastructure;
using Spectre.Console;
using Spectre.Console.Cli;

namespace Semantics.Gateway.Commands;

/// <summary>
/// semantics update — self-updates the gateway binary and pulls the latest container image.
/// </summary>
public sealed class UpdateCommand : Command<UpdateCommand.Settings>
{
    public sealed class Settings : CommandSettings { }

    private const string Repo = "famda/semantics";

    public override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken) {
        // Step 1: Self-update the gateway binary
        SelfUpdate();

        // Step 2: Pull the latest container image
        var image = DockerRunner.GetImage();
        DockerRunner.CheckDocker();

        var ok = DockerPuller.Pull(image, "Updating image");
        if (ok) {
            AnsiConsole.MarkupLine("[green]==> [/]Updated successfully.");
        } else {
            AnsiConsole.MarkupLine("[red]error:[/] Image update failed. Check your connection.");
        }

        return ok ? 0 : 1;
    }

    private static void SelfUpdate()
    {
        var currentVersion = GetCurrentVersion();
        var rid = GetRuntimeIdentifier();
        if (rid is null)
        {
            AnsiConsole.MarkupLine("[dim]Skipping CLI self-update (unsupported platform).[/]");
            return;
        }

        try
        {
            using var http = new HttpClient();
            http.DefaultRequestHeaders.UserAgent.ParseAdd("semantics-cli");

            var releaseUrl = $"https://api.github.com/repos/{Repo}/releases/latest";
            var json = http.GetStringAsync(releaseUrl).GetAwaiter().GetResult();
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var tagName = root.GetProperty("tag_name").GetString() ?? "";
            var remoteVersion = tagName.StartsWith("cli-v") ? tagName["cli-v".Length..] : tagName.TrimStart('v');

            if (!IsNewer(remoteVersion, currentVersion))
            {
                AnsiConsole.MarkupLine($"[dim]CLI is up to date (v{currentVersion}).[/]");
                return;
            }

            // Find the matching asset
            var assetName = $"semantics-{rid}" + (rid.StartsWith("win") ? ".exe" : "");
            string? downloadUrl = null;

            foreach (var asset in root.GetProperty("assets").EnumerateArray())
            {
                var name = asset.GetProperty("name").GetString();
                if (string.Equals(name, assetName, StringComparison.OrdinalIgnoreCase))
                {
                    downloadUrl = asset.GetProperty("browser_download_url").GetString();
                    break;
                }
            }

            if (downloadUrl is null)
            {
                AnsiConsole.MarkupLine($"[yellow]warn:[/] No release asset found for {rid}. Skipping CLI update.");
                return;
            }

            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("cyan"))
                .Start($"Updating CLI v{currentVersion} → v{remoteVersion} ...", _ =>
                {
                    var bytes = http.GetByteArrayAsync(downloadUrl).GetAwaiter().GetResult();

                    var currentExe = Environment.ProcessPath!;
                    var backupPath = currentExe + ".old";

                    // Rename current → .old, write new binary
                    if (File.Exists(backupPath))
                        File.Delete(backupPath);
                    File.Move(currentExe, backupPath);

                    try
                    {
                        File.WriteAllBytes(currentExe, bytes);

                        // Make executable on Unix
                        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                        {
                            Process.Start("chmod", $"+x \"{currentExe}\"")?.WaitForExit();
                        }

                        File.Delete(backupPath);
                    }
                    catch
                    {
                        // Rollback on failure
                        if (File.Exists(backupPath))
                            File.Move(backupPath, currentExe, overwrite: true);
                        throw;
                    }
                });

            AnsiConsole.MarkupLine($"[green]==> [/]CLI updated to v{remoteVersion}.");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[yellow]warn:[/] CLI self-update failed: {ex.Message}");
        }
    }

    private static string GetCurrentVersion()
    {
        var version = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion
            ?? Assembly.GetExecutingAssembly().GetName().Version?.ToString()
            ?? "0.0.0";

        var plusIndex = version.IndexOf('+');
        return plusIndex >= 0 ? version[..plusIndex] : version;
    }

    private static string? GetRuntimeIdentifier()
    {
        string os;
        string arch;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) os = "win";
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) os = "linux";
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) os = "osx";
        else return null;

        arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            _ => null!,
        };

        return arch is null ? null : $"{os}-{arch}";
    }

    private static bool IsNewer(string remote, string current)
    {
        if (Version.TryParse(remote, out var r) && Version.TryParse(current, out var c))
            return r > c;
        return false;
    }
}
