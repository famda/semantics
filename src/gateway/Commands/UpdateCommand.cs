using Semantics.Gateway.Infrastructure;
using Spectre.Console;
using Spectre.Console.Cli;

namespace Semantics.Gateway.Commands;

/// <summary>
/// semantics update — pulls the latest container image.
/// </summary>
public sealed class UpdateCommand : Command<UpdateCommand.Settings>
{
    public sealed class Settings : CommandSettings { }

    public override int Execute(CommandContext context, Settings settings)
    {
        var image = DockerRunner.GetImage();
        DockerRunner.CheckDocker();

        var ok = DockerPuller.Pull(image, "Updating");
        if (ok)
        {
            AnsiConsole.MarkupLine("[green]==> [/]Updated successfully.");
        }
        else
        {
            AnsiConsole.MarkupLine("[red]error:[/] Update failed. Check your connection.");
        }

        return ok ? 0 : 1;
    }
}
