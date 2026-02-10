using System.Reflection;
using Semantics.Gateway.Infrastructure;
using Spectre.Console;
using Spectre.Console.Cli;

namespace Semantics.Gateway.Commands;

/// <summary>
/// semantics version — shows gateway version and image info.
/// </summary>
public sealed class VersionCommand : Command<VersionCommand.Settings> {
    public sealed class Settings : CommandSettings { }

    public override int Execute(CommandContext context, Settings settings, CancellationToken cancellationToken) {

        var version = Assembly.GetExecutingAssembly()
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
            ?.InformationalVersion
            ?? Assembly.GetExecutingAssembly().GetName().Version?.ToString()
            ?? "unknown";

        // Strip build metadata (+sha) if present
        var plusIndex = version.IndexOf('+');
        if (plusIndex >= 0)
            version = version[..plusIndex];

        AnsiConsole.MarkupLine($"Semantics CLI v{version}");

        // Image info (only if Docker is available)
        var image = DockerRunner.GetImage();
        try {
            DockerRunner.CheckDocker();
            if (DockerRunner.HasImage(image)) {
                var digest = DockerRunner.GetImageDigest(image);
                AnsiConsole.MarkupLine($"[dim]Image:[/]   {image}");
                if (digest is not null)
                    AnsiConsole.MarkupLine($"[dim]Digest:[/]  {digest}");
            } else {
                AnsiConsole.MarkupLine($"[dim]Image:[/]   {image} [yellow](not pulled)[/]");
            }
        } catch {
            // Docker not available — just show the version
        }

        return 0;
    }
}
