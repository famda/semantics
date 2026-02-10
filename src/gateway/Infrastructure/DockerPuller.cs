using System.Diagnostics;
using Spectre.Console;

namespace Semantics.Gateway.Infrastructure;

/// <summary>
/// Silent Docker image pull with animated spinner.
/// </summary>
public static class DockerPuller
{
    /// <summary>
    /// Pull a Docker image with a Spectre.Console spinner.
    /// Returns true on success, false on failure.
    /// </summary>
    public static bool Pull(string image, string message = "Downloading")
    {
        return AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Style.Parse("cyan"))
            .Start($"{message} [bold]{image}[/] ...", _ =>
            {
                var psi = new ProcessStartInfo("docker")
                {
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };
                psi.ArgumentList.Add("pull");
                psi.ArgumentList.Add("-q");
                psi.ArgumentList.Add(image);

                using var proc = Process.Start(psi);
                if (proc is null) return false;

                proc.StandardOutput.ReadToEnd();
                proc.StandardError.ReadToEnd();
                proc.WaitForExit();

                return proc.ExitCode == 0;
            });
    }
}
