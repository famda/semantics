using Semantics.Gateway.Commands;
using Semantics.Gateway.Help;
using Spectre.Console.Cli;

// Intercept: no args or help flags → custom top-level help
if (args.Length == 0 || args[0] is "--help" or "-h" or "help")
{
    SemanticsHelpProvider.PrintTopLevelHelp();
    return 0;
}

// Pass-through subcommands: forward everything to Docker directly,
// bypassing Spectre.Console.Cli (which would intercept --help/-h).
var passThrough = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
{
    ["audio"] = "audio",
    ["video"] = "video",
    ["research"] = "research",
    ["docs"] = "docs",
};

if (passThrough.TryGetValue(args[0], out var subcommand))
{
    var remaining = args.Length > 1 ? args[1..] : [];
    return CliCommand.ExecutePassThrough(subcommand, remaining);
}

// Remaining commands: update, version — handled by Spectre.Console.Cli
var app = new CommandApp();

app.Configure(config =>
{
    config.SetApplicationName("semantics");
    config.Settings.HelpProviderStyles = null;

    config.AddCommand<UpdateCommand>("update")
          .WithDescription("Pull the latest container image");
    config.AddCommand<VersionCommand>("version")
          .WithDescription("Show CLI and image version");
});

return app.Run(args);
