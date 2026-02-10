namespace Semantics.Gateway.Infrastructure;

/// <summary>
/// Translates host paths (-i, -o, --config) into Docker volume mounts
/// and rewrites arguments to use the container-side paths.
/// </summary>
public static class PathResolver
{
    /// <summary>
    /// Parse command arguments, translate -i/-o/--config paths into volume
    /// mounts and container-relative arguments.
    /// </summary>
    public static (List<string> VolumeMounts, List<string> RewrittenArgs) Resolve(string[] args)
    {
        var volumeArgs = new List<string>();
        var rewrittenArgs = new List<string>();

        for (var i = 0; i < args.Length; i++)
        {
            var current = args[i];

            if (current is "-i" or "--input" && i + 1 < args.Length)
            {
                i++;
                var raw = args[i];

                // URL passthrough (video CLI supports YouTube URLs)
                if (raw.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
                    raw.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
                {
                    rewrittenArgs.Add(current);
                    rewrittenArgs.Add(raw);
                }
                else
                {
                    var resolved = Path.GetFullPath(raw);
                    if (!File.Exists(resolved))
                    {
                        Console.Error.WriteLine($"error: Input file not found: {raw}");
                        Environment.Exit(1);
                    }

                    var hostDir = Path.GetDirectoryName(resolved)!;
                    var fileName = Path.GetFileName(resolved);
                    volumeArgs.Add("-v");
                    volumeArgs.Add($"{hostDir}:/workspaces/input:ro");
                    rewrittenArgs.Add(current);
                    rewrittenArgs.Add($"/workspaces/input/{fileName}");
                }
            }
            else if (current is "-o" or "--output" && i + 1 < args.Length)
            {
                i++;
                var raw = args[i];
                var resolved = Path.GetFullPath(raw);

                if (!Directory.Exists(resolved))
                {
                    Directory.CreateDirectory(resolved);
                }

                volumeArgs.Add("-v");
                volumeArgs.Add($"{resolved}:/workspaces/output");
                rewrittenArgs.Add(current);
                rewrittenArgs.Add("/workspaces/output");
            }
            else if (current is "--config" && i + 1 < args.Length)
            {
                i++;
                var raw = args[i];
                var resolved = Path.GetFullPath(raw);

                if (!File.Exists(resolved))
                {
                    Console.Error.WriteLine($"error: Config file not found: {raw}");
                    Environment.Exit(1);
                }

                var hostDir = Path.GetDirectoryName(resolved)!;
                var fileName = Path.GetFileName(resolved);
                volumeArgs.Add("-v");
                volumeArgs.Add($"{hostDir}:/workspaces/config:ro");
                rewrittenArgs.Add(current);
                rewrittenArgs.Add($"/workspaces/config/{fileName}");
            }
            else
            {
                rewrittenArgs.Add(current);
            }
        }

        return (volumeArgs, rewrittenArgs);
    }
}
