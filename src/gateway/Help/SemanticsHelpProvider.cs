using Spectre.Console;

namespace Semantics.Gateway.Help;

/// <summary>
/// Custom top-level help renderer with box-drawing panels matching the
/// rich_click style used by the container CLIs.
/// </summary>
public static class SemanticsHelpProvider
{
    // Box-drawing characters
    private const char H   = '\u2500'; // ─
    private const char V   = '\u2502'; // │
    private const char TL  = '\u256D'; // ╭
    private const char TR  = '\u256E'; // ╮
    private const char BL  = '\u2570'; // ╰
    private const char BR  = '\u256F'; // ╯

    private const int BoxWidth = 70;

    public static void PrintTopLevelHelp()
    {
        var console = AnsiConsole.Console;

        console.WriteLine();
        console.Markup(" [yellow]Usage:[/] [cyan]semantics[/] <command> [[options]]");
        console.WriteLine();
        console.WriteLine();
        console.MarkupLine(" [dim]Semantics CLI \u2014 Unified interface for media intelligence[/]");
        console.MarkupLine(" [dim]Extract meaning, not just metadata. Composable AI operations designed for developers.[/]");
        console.WriteLine();

        // Commands panel
        WriteBoxTop("Commands", 59);
        WriteBoxRow("audio",    "Audio processing (transcription, diarization, ...)");
        WriteBoxRow("video",    "Video analysis (object detection, scenes, OCR, ...)");
        WriteBoxRow("research", "Web research (search, crawling, content extraction)");
        WriteBoxBottom();

        // Utility panel
        WriteBoxTop("Utility", 60);
        WriteBoxRow("update",  "Pull the latest container image");
        WriteBoxRow("version", "Show version information");
        WriteBoxRow("help",    "Show this help message");
        WriteBoxBottom();

        // Examples panel
        WriteBoxTop("Examples", 59);
        WriteBoxExample("semantics audio -i interview.mp4 -o ./results -t -d");
        WriteBoxExample("semantics video -i clip.mp4 -o ./results --from-segments -s -eo");
        WriteBoxExample("semantics research -o ./results -s 'AI trends' --download");
        WriteBoxBottom();

        console.WriteLine();
        console.MarkupLine(" [dim]Run[/] [cyan]semantics <command> --help[/] [dim]for command-specific options.[/]");
        console.WriteLine();
    }

    private static void WriteBoxTop(string title, int fillCount)
    {
        var fill = new string(H, fillCount);
        AnsiConsole.Console.WriteLine($"{TL}{H} {title} {fill}{TR}");
    }

    private static void WriteBoxBottom()
    {
        var fill = new string(H, BoxWidth);
        AnsiConsole.Console.WriteLine($"{BL}{fill}{BR}");
    }

    private static void WriteBoxRow(string name, string description)
    {
        var nameStr = name.PadRight(14);
        var descStr = description.PadRight(54);
        AnsiConsole.Console.Markup($"{V}  [green]{nameStr}[/]{descStr}{V}");
        AnsiConsole.Console.WriteLine();
    }

    private static void WriteBoxExample(string example)
    {
        var padded = example.PadRight(68);
        AnsiConsole.Console.WriteLine($"{V}  {padded}{V}");
    }
}
