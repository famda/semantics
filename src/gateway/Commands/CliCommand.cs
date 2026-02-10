using Semantics.Gateway.Infrastructure;
using Spectre.Console;

namespace Semantics.Gateway.Commands;

/// <summary>
/// Pass-through execution for audio/video/research subcommands.
/// Captures ALL remaining arguments and forwards them to Docker.
/// </summary>
public static class CliCommand
{
    /// <summary>
    /// Execute a pass-through subcommand by forwarding all args to Docker.
    /// Called directly from Program.cs, bypassing Spectre.Console.Cli routing
    /// so that flags like --help/-h are forwarded to the container CLI.
    /// </summary>
    public static int ExecutePassThrough(string subcommand, string[] rawArgs)
    {
        var image = DockerRunner.GetImage();

        DockerRunner.CheckDocker();

        if (!DockerRunner.HasImage(image))
        {
            var ok = DockerPuller.Pull(image, "Image not found \u2014 downloading");
            if (!ok)
            {
                AnsiConsole.MarkupLine("[red]error:[/] Failed to pull image. Check your connection.");
                return 1;
            }
        }

        var (volumeArgs, rewrittenArgs) = PathResolver.Resolve(rawArgs);
        var gpu = DockerRunner.DetectGpu();

        return DockerRunner.Run(image, subcommand, rewrittenArgs, volumeArgs, gpu);
    }
}
