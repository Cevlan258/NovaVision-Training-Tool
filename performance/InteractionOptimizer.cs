using System.Collections;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace NovaVision.Performance;

/// <summary>将耗时图片处理移出界面线程，并限制过期任务与缓存占用。</summary>
internal static class InteractionOptimizer
{
    private const long CacheLimit = 256L * 1024 * 1024;
    private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    private static readonly ConditionalWeakTable<object, State> States = new();
    private static readonly ConcurrentDictionary<(Type, string), MemberInfo> Members = new();
    private static readonly SemaphoreSlim DecodeSlots = new(2);
    private static readonly object CacheGate = new();
    private static readonly Dictionary<string, CacheEntry> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static long cacheBytes;
    private static long cacheClock;
    private static FileSystemWatcher? imageWatcher;
    private static string watchedDirectory = "";

    internal static bool TryApplyImageFilter(object vm)
    {
        if (Application.Current is null || !Application.Current.Dispatcher.CheckAccess()) return false;
        try
        {
            State state = States.GetValue(vm, _ => new State());
            string search = Get(vm, "SearchText") as string ?? "";
            object? category = Get(vm, "SelectedCategoryFilter");
            string status = Get(vm, "SelectedStatusFilter") as string ?? "";
            bool searchOnly = state.FilterInitialized && search != state.Search &&
                ReferenceEquals(category, state.FilterCategory) && status == state.FilterStatus;
            state.Search = search;
            state.FilterCategory = category;
            state.FilterStatus = status;
            state.FilterInitialized = true;
            state.FilterTimer?.Stop();
            if (!searchOnly)
            {
                Call(vm, "__PerfOriginal_ApplyImageFilter");
                return true;
            }
            state.FilterTimer ??= MakeTimer(150, () => Call(vm, "__PerfOriginal_ApplyImageFilter"));
            state.FilterTimer.Start();
            return true;
        }
        catch { return false; }
    }

    internal static bool TryApplyImageEnhancement(object vm)
    {
        if (Application.Current is null || !Application.Current.Dispatcher.CheckAccess()) return false;
        try
        {
            State state = States.GetValue(vm, _ => new State());
            state.EnhancementTimer?.Stop();
            state.EnhancementGeneration++;
            state.EnhancementTimer ??= MakeTimer(80, () => QueueEnhancement(vm, state));
            state.EnhancementTimer.Start();
            return true;
        }
        catch { return false; }
    }

    internal static bool TryLoadCurrentImage(object vm)
    {
        if (Application.Current is null || !Application.Current.Dispatcher.CheckAccess()) return false;
        try
        {
            State state = States.GetValue(vm, _ => new State());
            object? image = Get(vm, "CurrentImage");
            string? path = image is null ? null : Get(image, "FilePath") as string;
            EnsureImageWatcher(vm);
            state.LoadCancellation?.Cancel();
            state.LoadCancellation?.Dispose();
            state.LoadCancellation = new CancellationTokenSource();
            CancellationToken token = state.LoadCancellation.Token;
            long generation = ++state.LoadGeneration;
            state.EnhancementGeneration++;
            state.EnhancementTimer?.Stop();
            SetField(vm, "w", null);
            Set(vm, "CurrentImageBitmap", null);
            // 撤销基线在切图当下建立，迟到的解码不得清除用户刚做的修改。
            if (image is not null && path is not null && File.Exists(path))
            {
                Call(GetField(vm, "n")!, "Clear");
                Call(GetField(vm, "o")!, "Clear");
                Call(vm, "PushUndoState");
                Call(vm, "OnPropertyChanged", "CurrentFileNameDisplay");
                Call(vm, "OnPropertyChanged", "LabelCountDisplay");
                _ = LoadAndCommit(vm, state, image, path, generation, token);
            }
            return true;
        }
        catch { return false; }
    }

    private static async Task LoadAndCommit(object vm, State state, object image, string path,
        long generation, CancellationToken token)
    {
        try
        {
            BitmapSource bitmap = await Task.Run(() => Decode(path, token), token).ConfigureAwait(false);
            Dispatcher? dispatcher = Application.Current?.Dispatcher;
            if (dispatcher is null) return;
            await dispatcher.InvokeAsync(() =>
            {
                if (token.IsCancellationRequested || generation != state.LoadGeneration ||
                    !ReferenceEquals(image, Get(vm, "CurrentImage"))) return;
                SetField(vm, "w", bitmap);
                Set(image, "Width", (double)bitmap.PixelWidth);
                Set(image, "Height", (double)bitmap.PixelHeight);
                Set(vm, "CurrentImageBitmap", bitmap);
                QueueEnhancement(vm, state);
                PrefetchNeighbours(vm, image, generation, state, token);
            }, DispatcherPriority.Background);
        }
        catch (OperationCanceledException) { }
        catch (Exception exception)
        {
            System.Diagnostics.Trace.WriteLine(exception);
            Dispatcher? dispatcher = Application.Current?.Dispatcher;
            if (dispatcher is not null)
                await dispatcher.InvokeAsync(() =>
                {
                    if (generation == state.LoadGeneration && ReferenceEquals(image, Get(vm, "CurrentImage")))
                        Set(vm, "StatusText", "加载图像失败: " + exception.GetBaseException().Message);
                });
        }
    }

    private static void PrefetchNeighbours(object vm, object image, long generation, State state, CancellationToken token)
    {
        if (Get(vm, "Images") is not IList images) return;
        int index = images.IndexOf(image);
        var paths = new List<string>();
        foreach (int next in new[] { index - 1, index + 1 })
            if (next >= 0 && next < images.Count && Get(images[next]!, "FilePath") is string path && File.Exists(path))
                paths.Add(path);
        _ = Task.Run(() =>
        {
            foreach (string path in paths)
            {
                if (token.IsCancellationRequested || generation != state.LoadGeneration) break;
                try { _ = Decode(path, token); }
                catch (OperationCanceledException) { break; }
                catch { /* 预取失败留给实际打开图片时报告。 */ }
            }
        });
    }

    private static BitmapSource Decode(string path, CancellationToken token)
    {
        var file = new FileInfo(path);
        string key = Path.GetFullPath(path) + "|" + file.Length + "|" + file.LastWriteTimeUtc.Ticks;
        lock (CacheGate)
            if (Cache.TryGetValue(key, out CacheEntry? cached))
            {
                cached.LastUsed = ++cacheClock;
                return cached.Bitmap;
            }
        DecodeSlots.Wait(token);
        try
        {
            token.ThrowIfCancellationRequested();
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.StreamSource = stream;
            bitmap.EndInit();
            bitmap.Freeze();
            token.ThrowIfCancellationRequested();
            long bytes = checked((long)bitmap.PixelWidth * bitmap.PixelHeight * Math.Max(4, (bitmap.Format.BitsPerPixel + 7) / 8));
            lock (CacheGate)
            {
                // 文件发生变化时淘汰同路径的旧版本，避免旧解码无限留存。
                string prefix = Path.GetFullPath(path) + "|";
                foreach (string old in Cache.Keys.Where(k => k != key && k.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)).ToArray())
                {
                    cacheBytes -= Cache[old].Bytes;
                    Cache.Remove(old);
                }
                if (bytes <= CacheLimit && !Cache.ContainsKey(key))
                {
                    while (cacheBytes + bytes > CacheLimit && Cache.Count > 0)
                    {
                        string oldest = Cache.MinBy(pair => pair.Value.LastUsed).Key;
                        cacheBytes -= Cache[oldest].Bytes;
                        Cache.Remove(oldest);
                    }
                    Cache.Add(key, new CacheEntry(bitmap, bytes, ++cacheClock));
                    cacheBytes += bytes;
                }
            }
            return bitmap;
        }
        finally { DecodeSlots.Release(); }
    }

    private static void EnsureImageWatcher(object vm)
    {
        string? dataset = Get(vm, "CurrentDatasetPath") as string;
        if (string.IsNullOrWhiteSpace(dataset)) return;
        string directory = Path.Combine(Path.GetFullPath(dataset), "images");
        if (!Directory.Exists(directory)) return;
        lock (CacheGate)
        {
            if (StringComparer.OrdinalIgnoreCase.Equals(directory, watchedDirectory)) return;
            imageWatcher?.Dispose();
            Cache.Clear();
            cacheBytes = 0;
            watchedDirectory = directory;
            imageWatcher = new FileSystemWatcher(directory)
            {
                IncludeSubdirectories = true,
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.Size
            };
            imageWatcher.Changed += (_, e) => InvalidateImage(e.FullPath);
            imageWatcher.Created += (_, e) => InvalidateImage(e.FullPath);
            imageWatcher.Deleted += (_, e) => InvalidateImage(e.FullPath);
            imageWatcher.Renamed += (_, e) => { InvalidateImage(e.OldFullPath); InvalidateImage(e.FullPath); };
            imageWatcher.Error += (_, _) => { lock (CacheGate) { Cache.Clear(); cacheBytes = 0; } };
            imageWatcher.EnableRaisingEvents = true;
        }
    }

    private static void InvalidateImage(string path)
    {
        string prefix = Path.GetFullPath(path) + "|";
        lock (CacheGate)
            foreach (string key in Cache.Keys.Where(k => k.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)).ToArray())
            {
                cacheBytes -= Cache[key].Bytes;
                Cache.Remove(key);
            }
    }

    private static void QueueEnhancement(object vm, State state)
    {
        state.EnhancementTimer?.Stop();
        if (GetField(vm, "w") is not BitmapSource bitmap) return;
        object? image = Get(vm, "CurrentImage");
        double brightness = Convert.ToDouble(Get(vm, "Brightness"));
        double contrast = Convert.ToDouble(Get(vm, "Contrast"));
        bool clahe = Convert.ToBoolean(Get(vm, "EnableClahe"));
        long generation = ++state.EnhancementGeneration;
        if (Get(vm, "IsEnhanced") is false)
        {
            Set(vm, "CurrentImageBitmap", bitmap);
            return;
        }
        // 只有一个正在执行的增强，重复请求覆盖待处理项。
        state.PendingEnhancement = new EnhancementRequest(vm, image, bitmap, brightness, contrast, clahe, generation);
        if (state.EnhancementRunning) return;
        state.EnhancementRunning = true;
        _ = EnhanceLatest(state);
    }

    private static async Task EnhanceLatest(State state)
    {
        Dispatcher dispatcher = Application.Current.Dispatcher;
        try
        {
            while (true)
            {
                EnhancementRequest? request = await dispatcher.InvokeAsync(() =>
                {
                    EnhancementRequest? pending = state.PendingEnhancement;
                    state.PendingEnhancement = null;
                    return pending;
                });
                if (request is null) break;
                BitmapSource? result = await Task.Run(() =>
                {
                    // 用隔离的宿主对象执行原增强算法，保持CLAHE及颜色处理语义。
                    object shadow = RuntimeHelpers.GetUninitializedObject(request.Vm.GetType());
                    SetField(shadow, "w", request.Bitmap);
                    SetField(shadow, "x", request.Brightness);
                    SetField(shadow, "y", request.Contrast);
                    SetField(shadow, "z", request.Clahe);
                    Call(shadow, "__PerfOriginal_ApplyImageEnhancement");
                    var output = Get(shadow, "CurrentImageBitmap") as BitmapSource;
                    if (output is not null && output.CanFreeze && !output.IsFrozen) output.Freeze();
                    return output;
                }).ConfigureAwait(false);
                await dispatcher.InvokeAsync(() =>
                {
                    if (result is not null && request.Generation == state.EnhancementGeneration &&
                        ReferenceEquals(request.Image, Get(request.Vm, "CurrentImage")) &&
                        ReferenceEquals(request.Bitmap, GetField(request.Vm, "w")))
                        Set(request.Vm, "CurrentImageBitmap", result);
                });
            }
        }
        catch (Exception exception)
        {
            System.Diagnostics.Trace.WriteLine(exception);
            // 保留已显示的原图，下一次参数请求仍可恢复增强。
        }
        finally
        {
            await dispatcher.InvokeAsync(() =>
            {
                state.EnhancementRunning = false;
                if (state.PendingEnhancement is not null)
                {
                    state.EnhancementRunning = true;
                    _ = EnhanceLatest(state);
                }
            });
        }
    }

    internal static void AttachView(object value)
    {
        if (value is not FrameworkElement view) return;
        view.Loaded += (_, _) => ApplyVirtualization(view);
        view.DataContextChanged += (_, _) => ApplyVirtualization(view);
        ApplyVirtualization(view);
    }

    private static void ApplyVirtualization(FrameworkElement view)
    {
        if (view.DataContext is not object vm) return;
        object? images;
        object? collectionView;
        try { images = Get(vm, "Images"); collectionView = Get(vm, "ImagesView"); }
        catch { return; }
        Visit(view);
        void Visit(DependencyObject node)
        {
            if (node is ListBox list && (ReferenceEquals(list.ItemsSource, images) ||
                ReferenceEquals(list.ItemsSource, collectionView) ||
                list.ItemsSource is ICollectionView cv && ReferenceEquals(cv.SourceCollection, images)))
            {
                VirtualizingPanel.SetIsVirtualizing(list, true);
                VirtualizingPanel.SetIsVirtualizingWhenGrouping(list, true);
                VirtualizingPanel.SetVirtualizationMode(list, VirtualizationMode.Recycling);
                ScrollViewer.SetCanContentScroll(list, true);
                list.SetCurrentValue(ItemsControl.ItemsPanelProperty,
                    new ItemsPanelTemplate(new FrameworkElementFactory(typeof(VirtualizingStackPanel))));
            }
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(node); i++) Visit(VisualTreeHelper.GetChild(node, i));
        }
    }

    private static DispatcherTimer MakeTimer(int milliseconds, Action action)
    {
        var timer = new DispatcherTimer(DispatcherPriority.Background) { Interval = TimeSpan.FromMilliseconds(milliseconds) };
        timer.Tick += (_, _) =>
        {
            timer.Stop();
            try { action(); }
            catch (Exception error) { System.Diagnostics.Debug.WriteLine(error); }
        };
        return timer;
    }

    private static object? Get(object target, string name) =>
        ((PropertyInfo)Members.GetOrAdd((target.GetType(), "P:" + name), _ => target.GetType().GetProperty(name, Flags)!)).GetValue(target);
    private static void Set(object target, string name, object? value) =>
        ((PropertyInfo)Members.GetOrAdd((target.GetType(), "P:" + name), _ => target.GetType().GetProperty(name, Flags)!)).SetValue(target, value);
    private static object? GetField(object target, string name) =>
        ((FieldInfo)Members.GetOrAdd((target.GetType(), "F:" + name), _ => target.GetType().GetField(name, Flags)!)).GetValue(target);
    private static void SetField(object target, string name, object? value) =>
        ((FieldInfo)Members.GetOrAdd((target.GetType(), "F:" + name), _ => target.GetType().GetField(name, Flags)!)).SetValue(target, value);
    private static object? Call(object target, string name, params object?[] args)
    {
        Type type = target.GetType();
        var method = (MethodInfo)Members.GetOrAdd((type, "M:" + name), _ => type.GetMethod(name, Flags)!);
        return method.Invoke(target, args);
    }

    private sealed class State
    {
        internal DispatcherTimer? FilterTimer;
        internal bool FilterInitialized;
        internal string Search = "";
        internal object? FilterCategory;
        internal string FilterStatus = "";
        internal CancellationTokenSource? LoadCancellation;
        internal long LoadGeneration;
        internal DispatcherTimer? EnhancementTimer;
        internal long EnhancementGeneration;
        internal bool EnhancementRunning;
        internal EnhancementRequest? PendingEnhancement;
    }

    private sealed record EnhancementRequest(object Vm, object? Image, BitmapSource Bitmap,
        double Brightness, double Contrast, bool Clahe, long Generation);
    private sealed class CacheEntry(BitmapSource bitmap, long bytes, long lastUsed)
    {
        internal readonly BitmapSource Bitmap = bitmap;
        internal readonly long Bytes = bytes;
        internal long LastUsed = lastUsed;
    }
}
