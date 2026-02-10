using System.Diagnostics;

namespace Semantics.Gateway.Infrastructure;

/// <summary>
/// Core Docker execution engine — process spawning, GPU detection, signal handling.
/// </summary>
public static class DockerRunner
{
    private static readonly string[] EnvVars =
    [
        "TF_ENABLE_ONEDNN_OPTS=0",
        "TF_DISABLE_XLA=1",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
    ];

    /// <summary>
    /// Verify that Docker is installed and the daemon is running.
    /// Exits with a clear message if not.
    /// </summary>
    public static void CheckDocker()
    {
        try
        {
            var psi = new ProcessStartInfo("docker", "info")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi)!;
            proc.StandardOutput.ReadToEnd();
            proc.StandardError.ReadToEnd();
            proc.WaitForExit();

            if (proc.ExitCode != 0)
            {
                Console.Error.WriteLine("error: Docker daemon is not running. Please start Docker Desktop and try again.");
                Environment.Exit(1);
            }
        }
        catch (Exception)
        {
            Console.Error.WriteLine("error: Docker is not installed. Install: https://docs.docker.com/get-docker/");
            Environment.Exit(1);
        }
    }

    /// <summary>
    /// Check whether a Docker image exists locally.
    /// </summary>
    public static bool HasImage(string image)
    {
        try
        {
            var psi = new ProcessStartInfo("docker", $"image inspect {image}")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi)!;
            proc.StandardOutput.ReadToEnd();
            proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            return proc.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Detect NVIDIA GPU support by parsing `docker info` output.
    /// </summary>
    public static bool DetectGpu()
    {
        try
        {
            var psi = new ProcessStartInfo("docker", "info")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi)!;
            var output = proc.StandardOutput.ReadToEnd();
            proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            return output.Contains("nvidia", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Get the short digest of a local Docker image (first 19 chars).
    /// Returns null if the image is not found.
    /// </summary>
    public static string? GetImageDigest(string image)
    {
        try
        {
            var psi = new ProcessStartInfo("docker", $"image inspect --format {{{{.Id}}}} {image}")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi)!;
            var output = proc.StandardOutput.ReadToEnd().Trim();
            proc.StandardError.ReadToEnd();
            proc.WaitForExit();

            if (proc.ExitCode != 0 || string.IsNullOrEmpty(output))
                return null;

            return output.Length > 19 ? output[..19] : output;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Execute a Docker container with full TTY passthrough and signal forwarding.
    /// </summary>
    public static int Run(
        string image,
        string subcommand,
        IReadOnlyList<string> rewrittenArgs,
        IReadOnlyList<string> volumeArgs,
        bool gpu)
    {
        // Build the inner bash command: semantics-{sub} arg1 arg2 ...
        var bashCmd = $"semantics-{subcommand}";
        foreach (var arg in rewrittenArgs)
        {
            bashCmd += " " + EscapeBashArg(arg);
        }

        // Build docker run arguments
        var args = new List<string> { "run", "--rm", "--init" };

        // TTY + interactive — enables Ctrl+C signal forwarding
        if (!Console.IsOutputRedirected && !Console.IsInputRedirected)
        {
            args.Add("-it");
        }

        // GPU support
        if (gpu)
        {
            args.Add("--gpus");
            args.Add("all");
        }

        // Environment variables
        foreach (var env in EnvVars)
        {
            args.Add("-e");
            args.Add(env);
        }

        // Volume mounts
        foreach (var vol in volumeArgs)
        {
            args.Add(vol);
        }

        // Image + entrypoint
        args.Add(image);
        args.Add("-lc");
        args.Add(bashCmd);

        var psi = new ProcessStartInfo("docker")
        {
            UseShellExecute = false,
            // Do NOT redirect — full TTY passthrough
        };

        foreach (var arg in args)
        {
            psi.ArgumentList.Add(arg);
        }

        using var process = Process.Start(psi);
        if (process is null)
        {
            Console.Error.WriteLine("error: Failed to start Docker process.");
            return 1;
        }

        // Ctrl+C handling: suppress in the gateway so Docker forwards SIGINT
        // to the container via --init. A second Ctrl+C falls through for hard kill.
        var ctrlCCount = 0;
        Console.CancelKeyPress += (_, e) =>
        {
            ctrlCCount++;
            if (ctrlCCount == 1)
            {
                e.Cancel = true; // Let Docker/init handle the first SIGINT
            }
            // Second press: e.Cancel defaults to false → process tree killed
        };

        process.WaitForExit();
        return process.ExitCode;
    }

    /// <summary>
    /// Escape an argument for use inside bash -lc "...".
    /// Simple args pass through; anything with special chars gets single-quoted.
    /// </summary>
    private static string EscapeBashArg(string arg)
    {
        // Simple args: alphanumeric, dots, slashes, colons, dashes, underscores
        if (arg.All(c => char.IsLetterOrDigit(c) || c is '.' or '/' or ':' or '-' or '_' or '='))
        {
            return arg;
        }

        // Wrap in single quotes, escape embedded single quotes for bash
        return "'" + arg.Replace("'", "'\\''") + "'";
    }

    /// <summary>
    /// Get the configured Docker image name.
    /// </summary>
    public static string GetImage()
    {
        return Environment.GetEnvironmentVariable("SEMANTICS_IMAGE")
               ?? "famda/semantics:cli-latest";
    }
}
