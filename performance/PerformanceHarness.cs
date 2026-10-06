using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.Loader;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

internal static class PerformanceHarness
{
    private static Assembly app = null!;
    private static string appDirectory = "";
    private static object? diagnosticVm;
    private const BindingFlags All = BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;
    private static object Create(string name) => Activator.CreateInstance(app.GetType("ViTooAI." + name, true)!)!;
    private static object? Get(object value, string name) => value.GetType().GetProperty(name, All)!.GetValue(value);
    private static void Set(object value, string name, object? data) => value.GetType().GetProperty(name, All)!.SetValue(value, data);
    private static object? Call(object value, string name, params object?[] args) => value.GetType().GetMethod(name, All)!.Invoke(value, args);
    private static void Check(bool value, string message) { if (!value) throw new InvalidOperationException(message); }

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            Trace.Listeners.Add(new ConsoleTraceListener());
            appDirectory = Path.GetFullPath(args[0]);
            if (args.Length > 1 && args[1] == "loader")
            {
                // 此模式依赖应用deps清单进行默认加载，不安装自定义解析事件。
                app = Assembly.Load("NovaVision");
                _ = new Application();
                object instance = Create("ViewModels.AnnotateViewModel");
                Call(instance, "UpdateCategoryCounts");
                string loaded = AssemblyLoadContext.Default.Assemblies.Single(a => a.GetName().Name == "NovaVision.Performance").FullName!;
                Check((string)Get(instance, "AnnotatedCountDisplay")! == "已标注: 0 / 0 张图像", "默认加载后的方法结果错误");
                Console.WriteLine("Default ALC/deps PASS: " + loaded);
                return 0;
            }
            AssemblyLoadContext.Default.Resolving += (_, name) =>
            {
                string path = Path.Combine(appDirectory, name.Name + ".dll");
                return File.Exists(path) ? AssemblyLoadContext.Default.LoadFromAssemblyPath(path) : null;
            };
            AssemblyLoadContext.Default.ResolvingUnmanagedDll += (_, name) =>
            {
                foreach (string directory in new[] { appDirectory, Path.Combine(appDirectory, "runtimes", "win-x64", "native") })
                {
                    string path = Path.Combine(directory, name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) ? name : name + ".dll");
                    if (File.Exists(path)) return System.Runtime.InteropServices.NativeLibrary.Load(path);
                }
                return IntPtr.Zero;
            };
            app = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.Combine(appDirectory, "NovaVision.dll"));
            // 验证进程的入口是harness，显式为WPF资源解析指定宿主程序集。
            typeof(Application).GetField("_resourceAssembly", BindingFlags.Static | BindingFlags.NonPublic)?.SetValue(null, app);
            // 只加载应用资源，取消构造函数排队的启动事件，避免测试开启真实主窗口。
            var startupOperations = new List<DispatcherOperation>();
            DispatcherHookEventHandler posted = (_, e) => startupOperations.Add(e.Operation);
            Dispatcher.CurrentDispatcher.Hooks.OperationPosted += posted;
            var application = (Application)Activator.CreateInstance(app.GetType("ViTooAI.App", true)!)!;
            Dispatcher.CurrentDispatcher.Hooks.OperationPosted -= posted;
            foreach (DispatcherOperation operation in startupOperations) operation.Abort();
            application.GetType().GetMethod("InitializeComponent", All)!.Invoke(application, null);
            if (args.Length < 2 || args[1] == "inspect")
            {
                foreach (string typeName in new[] { "Models.ImageItem", "Models.BoundingBox", "Models.CategoryItem", "ViewModels.AnnotateViewModel" })
                {
                    Type type = app.GetType("ViTooAI." + typeName, true)!;
                    Console.WriteLine(typeName + ": " + string.Join(", ", type.GetProperties(All).Select(p => p.Name + "=" + p.PropertyType.Name)));
                }
                object empty = Create("ViewModels.AnnotateViewModel");
                Console.WriteLine("VM constructed: " + Get(empty, "AnnotatedCountDisplay"));
                foreach (MethodInfo method in app.GetType("ViTooAI.Services.YoloDatasetService", true)!.GetMethods(All).Where(m => m.Name is "LoadYoloAnnotations" or "SaveYoloAnnotations"))
                    Console.WriteLine(method.Name + ": " + string.Join(", ", method.GetParameters().Select(p => p.Name + "=" + p.ParameterType.Name)));
                return 0;
            }
            if (args[1] == "counts") return Counts(args);
            if (args[1] == "save") return Save(args);
            if (args[1] == "ui") return Ui(args);
            if (args[1] == "contracts") return Contracts(args);
            throw new ArgumentException("未知验证模式");
        }
        catch (Exception exception)
        {
            if (diagnosticVm is not null)
                Console.Error.WriteLine("状态: " + Get(diagnosticVm, "StatusText") + "; 当前: " + Get(Get(diagnosticVm, "CurrentImage")!, "FilePath"));
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static int Counts(string[] args)
    {
        int count = int.Parse(args[2]);
        object vm = Create("ViewModels.AnnotateViewModel");
        IList images = (IList)Get(vm, "Images")!;
        IList categories = (IList)Get(vm, "Categories")!;
        categories.Clear();
        for (int c = 0; c < 5; c++)
        {
            object category = Create("Models.CategoryItem");
            Set(category, "Id", c);
            Set(category, "Name", "class" + c);
            categories.Add(category);
        }
        for (int i = 0; i < count; i++)
        {
            object image = Create("Models.ImageItem");
            Set(image, "FilePath", Path.Combine(Path.GetTempPath(), "novavision-counts", i + ".png"));
            Set(image, "RelativePath", i + ".png");
            object box = Create("Models.BoundingBox");
            Set(box, "ClassId", i % 5);
            ((IList)Get(image, "BoundingBoxes")!).Add(box);
            Call(image, "UpdateAnnotationStatus");
            images.Add(image);
        }
        Call(vm, "UpdateCategoryCounts");
        VerifyCounts(vm);
        var measurements = new List<double>();
        // 测量真实程序集方法，不用模拟耗时代替实测。
        for (int round = 0; round < 60; round++)
        {
            object changed = images[round % images.Count]!;
            IList boxes = (IList)Get(changed, "BoundingBoxes")!;
            Set(boxes[0]!, "ClassId", (round + 1) % 5);
            Stopwatch timer = Stopwatch.StartNew();
            Call(vm, "UpdateCategoryCounts");
            _ = Get(vm, "AnnotatedCountDisplay");
            _ = Get(vm, "ImageListHeaderDisplay");
            timer.Stop();
            measurements.Add(timer.Elapsed.TotalMilliseconds);
        }
        VerifyCounts(vm);
        object first = images[0]!;
        ((IList)Get(first, "BoundingBoxes")!).Clear();
        Set(first, "IsNegativeSample", true);
        Call(first, "UpdateAnnotationStatus");
        Call(vm, "UpdateCategoryCounts");
        VerifyCounts(vm);
        Check((bool)Get(first, "IsAnnotated")!, "负样本应属于已标注图片");
        Set(first, "IsNegativeSample", false);
        Call(first, "UpdateAnnotationStatus");
        Call(vm, "UpdateCategoryCounts");
        VerifyCounts(vm);
        images.RemoveAt(1);
        Call(vm, "UpdateCategoryCounts");
        VerifyCounts(vm);
        measurements.Sort();
        var report = new { mode = "counts", dataset_kind = "constructed_objects", images = count,
            median_ms = measurements[measurements.Count / 2], p95_ms = measurements[(int)(measurements.Count * 0.95)], checks = "class edits, negative samples, clear boxes, remove image" };
        string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
        if (args.Length > 3) File.WriteAllText(args[3], json);
        Console.WriteLine(json);
        return 0;
    }

    private static void VerifyCounts(object vm)
    {
        var actual = new Dictionary<int, int>();
        int annotated = 0;
        foreach (object image in (IEnumerable)Get(vm, "Images")!)
        {
            if ((bool)Get(image, "IsAnnotated")!) annotated++;
            foreach (object box in (IEnumerable)Get(image, "BoundingBoxes")!)
            {
                int key = (int)Get(box, "ClassId")!;
                actual[key] = actual.GetValueOrDefault(key) + 1;
            }
        }
        foreach (object category in (IEnumerable)Get(vm, "Categories")!)
            Check((int)Get(category, "Count")! == actual.GetValueOrDefault((int)Get(category, "Id")!), "类别计数与独立重算不一致");
        string display = (string)Get(vm, "AnnotatedCountDisplay")!;
        Check(display.Contains($"已标注: {annotated} /"), "已标注显示与独立重算不一致: " + display);
    }

    private static int Save(string[] args)
    {
        // 只接受显式准备的隔离测试目录，禁止对用户数据调用保存。
        string dataset = Path.GetFullPath(args[2]);
        Check(File.Exists(Path.Combine(dataset, ".novavision-performance-fixture")), "缺少隔离测试目录标记");
        object vm = Create("ViewModels.AnnotateViewModel");
        Set(vm, "CurrentDatasetPath", dataset);
        object service = Create("Services.YoloDatasetService");
        MethodInfo load = service.GetType().GetMethod("LoadImagesFromFolder", All)!;
        var parameters = load.GetParameters();
        Console.WriteLine("LoadImagesFromFolder: " + string.Join(", ", parameters.Select(p => p.Name + "=" + p.ParameterType.Name)));
        object?[] inputs = parameters.Select(p => p.HasDefaultValue ? p.DefaultValue : null).ToArray();
        inputs[0] = Path.Combine(dataset, "images");
        if (inputs.Length > 1) inputs[1] = Activator.CreateInstance(parameters[1].ParameterType);
        object loaded = load.Invoke(service, inputs)!;
        IList categories = (IList)Get(vm, "Categories")!;
        categories.Clear();
        if (inputs.Length > 1 && inputs[1] is IEnumerable loadedCategories)
            foreach (object category in loadedCategories) categories.Add(category);
        if (categories.Count == 0)
            for (int i = 0; i < 4; i++)
            {
                object category = Create("Models.CategoryItem");
                Set(category, "Id", i);
                Set(category, "Name", new[] { "D00", "D10", "D20", "D40" }[i]);
                categories.Add(category);
            }
        IList images = (IList)Get(vm, "Images")!;
        foreach (object image in (IEnumerable)loaded)
        {
            LoadAnnotations(service, image, inputs[1]!);
            images.Add(image);
        }
        Check(images.Count > 0, "隔离样本为空");
        object current = images.Cast<object>().First(image => ((IList)Get(image, "BoundingBoxes")!).Count > 0);
        Set(vm, "CurrentImage", current);
        WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "保存测试图片未加载");
        Call(vm, "SaveCurrentAnnotations");
        VerifyCounts(vm);
        Console.WriteLine("save task=" + Get(vm, "CurrentYoloTaskType") + " categories=" + categories.Count + " status=" + Get(vm, "StatusText"));
        var measurements = new List<double>();
        string imagePath = (string)Get(current, "FilePath")!;
        string labelPath = Path.ChangeExtension(imagePath.Replace(Path.DirectorySeparatorChar + "images" + Path.DirectorySeparatorChar,
            Path.DirectorySeparatorChar + "labels" + Path.DirectorySeparatorChar), ".txt");
        byte[]? originalLabel = File.Exists(labelPath) ? File.ReadAllBytes(labelPath) : null;
        string classesPath = Path.Combine(dataset, "classes.txt");
        string yamlPath = Path.Combine(dataset, "data.yaml");
        DateTime labelTime = File.Exists(labelPath) ? File.GetLastWriteTimeUtc(labelPath) : DateTime.MinValue;
        DateTime classesTime = File.GetLastWriteTimeUtc(classesPath);
        DateTime yamlTime = File.GetLastWriteTimeUtc(yamlPath);
        for (int i = 0; i < 20; i++)
        {
            Stopwatch timer = Stopwatch.StartNew();
            Call(vm, "SaveCurrentAnnotations");
            timer.Stop();
            measurements.Add(timer.Elapsed.TotalMilliseconds);
        }
        measurements.Sort();
        bool optimized = vm.GetType().GetMethod("__PerfOriginal_SaveCurrentAnnotations", All) is not null;
        bool unchangedWritesSkipped = (!File.Exists(labelPath) || labelTime == File.GetLastWriteTimeUtc(labelPath)) &&
            classesTime == File.GetLastWriteTimeUtc(classesPath) && yamlTime == File.GetLastWriteTimeUtc(yamlPath);
        if (optimized) Check(unchangedWritesSkipped, "未修改标注时仍重复写文件");
        IList boxes = (IList)Get(current, "BoundingBoxes")!;
        if (boxes.Count > 0)
        {
            Set(boxes[0]!, "ClassId", ((int)Get(boxes[0]!, "ClassId")! + 1) % categories.Count);
            Call(vm, "SaveCurrentAnnotations");
            string expected = string.Concat(boxes.Cast<object>().Select(box => (string)Call(box, "ToYoloLine")! + Environment.NewLine));
            Check(File.ReadAllText(labelPath) == expected, "编辑后的标签未写入");
            byte[] beforeFailure = File.ReadAllBytes(labelPath);
            Set(boxes[0]!, "ClassId", ((int)Get(boxes[0]!, "ClassId")! + 1) % categories.Count);
            using (var blocked = new FileStream(labelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                Call(vm, "SaveCurrentAnnotations");
                Check(File.ReadAllBytes(labelPath).SequenceEqual(beforeFailure), "写入阻塞时标签异常变化");
            }
            Call(vm, "SaveCurrentAnnotations");
            expected = string.Concat(boxes.Cast<object>().Select(box => (string)Call(box, "ToYoloLine")! + Environment.NewLine));
            Check(File.ReadAllText(labelPath) == expected, "写入失败解除后没有重试保存");
        }
        // 外部配置变化不得被缓存掩盖，即使恢复长度和时间戳也应核验内容。
        DateTime previousClassTime = File.GetLastWriteTimeUtc(classesPath);
        byte[] previousClasses = File.ReadAllBytes(classesPath);
        byte[] changedClasses = (byte[])previousClasses.Clone();
        changedClasses[0] = changedClasses[0] == (byte)'Z' ? (byte)'X' : (byte)'Z';
        File.WriteAllBytes(classesPath, changedClasses);
        File.SetLastWriteTimeUtc(classesPath, previousClassTime);
        Call(vm, "SaveCurrentAnnotations");
        Check(File.ReadAllBytes(classesPath).SequenceEqual(previousClasses), "外部同长同时间类别文件变化没有修复");
        string savedYaml = File.ReadAllText(yamlPath);
        File.AppendAllText(yamlPath, "\n# external-change\n");
        Call(vm, "SaveCurrentAnnotations");
        Check(File.ReadAllText(yamlPath) == savedYaml, "外部yaml变化未触发原元数据同步");
        boxes.Clear();
        Set(current, "IsNegativeSample", true);
        Call(vm, "SaveCurrentAnnotations");
        Check(File.Exists(labelPath) && new FileInfo(labelPath).Length == 0 && (bool)Get(current, "IsAnnotated")!, "负样本保存语义错误");
        Set(current, "IsNegativeSample", false);
        Call(vm, "SaveCurrentAnnotations");
        Check(!File.Exists(labelPath), "清除负样本后空标签未删除");
        Set(categories[0]!, "Name", "temporary-perf-name");
        Call(vm, "SaveCurrentAnnotations");
        Check(File.ReadAllText(classesPath).Contains("temporary-perf-name") && File.ReadAllText(yamlPath).Contains("temporary-perf-name"), "类别改名后配置未同步");
        Set(categories[0]!, "Name", "D00");
        Call(vm, "SaveCurrentAnnotations");
        if (originalLabel is not null) File.WriteAllBytes(labelPath, originalLabel);
        string json = JsonSerializer.Serialize(new { mode = "save", images = images.Count, median_ms = measurements[10], p95_ms = measurements[19], dataset,
            unchanged_writes_skipped = unchangedWritesSkipped, checks = "edit label, failed save retry, external metadata, negative sample, delete empty label, rename class and sync yaml" }, new JsonSerializerOptions { WriteIndented = true });
        if (args.Length > 3) File.WriteAllText(args[3], json);
        Console.WriteLine(json);
        return 0;
    }

    private static void Pump(int milliseconds)
    {
        var frame = new DispatcherFrame();
        var timer = new DispatcherTimer(DispatcherPriority.Send) { Interval = TimeSpan.FromMilliseconds(milliseconds) };
        timer.Tick += (_, _) => { timer.Stop(); frame.Continue = false; };
        timer.Start();
        Dispatcher.PushFrame(frame);
    }

    private static void WaitFor(Func<bool> condition, string message)
    {
        Stopwatch timer = Stopwatch.StartNew();
        while (!condition() && timer.Elapsed.TotalSeconds < 15) Pump(10);
        Check(condition(), message);
    }

    private static int Ui(string[] args)
    {
        string dataset = Path.GetFullPath(args[2]);
        Check(File.Exists(Path.Combine(dataset, ".novavision-performance-fixture")), "缺少隔离测试目录标记");
        object vm = Create("ViewModels.AnnotateViewModel");
        diagnosticVm = vm;
        Set(vm, "CurrentDatasetPath", dataset);
        object service = Create("Services.YoloDatasetService");
        MethodInfo loadImages = service.GetType().GetMethod("LoadImagesFromFolder", All)!;
        object?[] inputs = { Path.Combine(dataset, "images"), Activator.CreateInstance(loadImages.GetParameters()[1].ParameterType) };
        object loaded = loadImages.Invoke(service, inputs)!;
        IList images = (IList)Get(vm, "Images")!;
        foreach (object image in (IEnumerable)loaded)
        {
            LoadAnnotations(service, image, inputs[1]!);
            images.Add(image);
        }
        Check(images.Count > 2, "需要至少三张隔离样本");
        Set(vm, "AutoSaveEnabled", false);
        Set(vm, "KeepPreviousAnnotations", false);
        object a = images[0]!;
        object b = images[1]!;
        object c = images[2]!;
        double defaultContrast = (double)Get(vm, "Contrast")!;
        bool optimized = vm.GetType().GetMethod("__PerfOriginal_z", All) is not null;
        Set(vm, "CurrentImage", a);
        Set(vm, "CurrentImage", b);
        Set(vm, "CurrentImage", c);
        WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "最终图片未完成加载");
        Pump(200);
        Check(ReferenceEquals(Get(vm, "CurrentImage"), c), "异步结果改错当前图片");
        var rendered = (BitmapSource)Get(vm, "CurrentImageBitmap")!;
        using (var stream = File.OpenRead((string)Get(c, "FilePath")!))
        {
            var actual = BitmapDecoder.Create(stream, BitmapCreateOptions.None, BitmapCacheOption.OnLoad).Frames[0];
            Check(rendered.PixelWidth == actual.PixelWidth && rendered.PixelHeight == actual.PixelHeight, "晚到解码尺寸覆盖当前图片");
            Check((double)Get(c, "Width")! == actual.PixelWidth && (double)Get(c, "Height")! == actual.PixelHeight, "模型尺寸未与显示对齐");
        }
        if (optimized)
        {
            // 主动阻塞解码槽位，使取消分支必然发生，而不是依赖图片偶然较慢。
            Assembly helper = AssemblyLoadContext.Default.Assemblies.Single(assembly => assembly.GetName().Name == "NovaVision.Performance");
            Type interactions = helper.GetType("NovaVision.Performance.InteractionOptimizer", true)!;
            var slots = (SemaphoreSlim)interactions.GetField("DecodeSlots", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null)!;
            slots.Wait(); slots.Wait();
            try
            {
                Set(vm, "CurrentImage", images[40]);
                Set(vm, "CurrentImage", images[41]);
                Set(vm, "CurrentImage", images[42]);
            }
            finally { slots.Release(2); }
            object latest = images[42]!;
            IList currentBoxes = (IList)Get(latest, "BoundingBoxes")!;
            int priorCount = currentBoxes.Count;
            object extra = Create("Models.BoundingBox");
            Set(extra, "ClassId", 0);
            Call(vm, "PushUndoState");
            currentBoxes.Add(extra);
            WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "受控取消后的最新图没有完成");
            Check(ReferenceEquals(Get(vm, "CurrentImage"), latest), "取消后身份错误");
            Call(vm, "Undo");
            Check(((IList)Get(latest, "BoundingBoxes")!).Count == priorCount, "解码完成清除了刚编辑的撤销基线");
            Call(vm, "Redo");
            Check(((IList)Get(latest, "BoundingBoxes")!).Count == priorCount + 1, "解码后重做错误");
            // 让已完成解码的提交停留在Dispatcher队列中，再取消它。
            using var postedFromWorker = new ManualResetEventSlim();
            int uiThread = Environment.CurrentManagedThreadId;
            DispatcherHookEventHandler postedCallback = (_, e) =>
            {
                if (Environment.CurrentManagedThreadId != uiThread && e.Operation.Priority == DispatcherPriority.Background)
                    postedFromWorker.Set();
            };
            Dispatcher.CurrentDispatcher.Hooks.OperationPosted += postedCallback;
            Set(vm, "CurrentImage", a);
            bool obsoleteReady = postedFromWorker.Wait(5000);
            Dispatcher.CurrentDispatcher.Hooks.OperationPosted -= postedCallback;
            Check(obsoleteReady, "未能确定性排队旧图片提交");
            Set(vm, "CurrentImage", b);
            WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "旧提交取消后的新图片未加载");
            Check(ReferenceEquals(Get(vm, "CurrentImage"), b), "迟到结果改变了图片身份");
            Set(vm, "CurrentImage", c);
            WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "回到增强测试图片失败");
        }
        // 继承上一图标注与各任务导航使用同一套原公开命令。
        object inheritanceSource = images.Cast<object>().First(image => ((IList)Get(image, "BoundingBoxes")!).Count > 0);
        int sourceIndex = images.IndexOf(inheritanceSource);
        object inheritanceTarget = images[(sourceIndex + 1) % images.Count]!;
        ((IList)Get(inheritanceTarget, "BoundingBoxes")!).Clear();
        Set(inheritanceTarget, "IsNegativeSample", false);
        Call(inheritanceTarget, "UpdateAnnotationStatus");
        Set(vm, "CurrentImage", inheritanceSource);
        WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "继承源图片未加载");
        int sourceBoxes = ((IList)Get(inheritanceSource, "BoundingBoxes")!).Count;
        Set(vm, "KeepPreviousAnnotations", true);
        Set(vm, "CurrentImage", inheritanceTarget);
        WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "继承路径图片未加载");
        Check(((IList)Get(inheritanceTarget, "BoundingBoxes")!).Count == sourceBoxes, "上一图标注未正确继承");
        Set(vm, "CurrentImage", c);
        WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "继承路径导航失败");
        Set(vm, "KeepPreviousAnnotations", false);
        // 参数连续变更后必须提交最后一组参数处理的结果。
        Set(vm, "Brightness", 10.0);
        Set(vm, "Brightness", 20.0);
        Set(vm, "Contrast", 1.2);
        Pump(1000);
        Check(Get(vm, "CurrentImageBitmap") is BitmapSource, "增强后图片丢失");
        object shadow = RuntimeHelpers.GetUninitializedObject(vm.GetType());
        BitmapSource? originalBitmap = vm.GetType().GetField("w", All)!.GetValue(vm) as BitmapSource;
        vm.GetType().GetField("w", All)!.SetValue(shadow, originalBitmap);
        vm.GetType().GetField("x", All)!.SetValue(shadow, 20.0);
        vm.GetType().GetField("y", All)!.SetValue(shadow, 1.2);
        vm.GetType().GetField("z", All)!.SetValue(shadow, false);
        string algorithm = vm.GetType().GetMethod("__PerfOriginal_ApplyImageEnhancement", All) is not null
            ? "__PerfOriginal_ApplyImageEnhancement" : "ApplyImageEnhancement";
        Call(shadow, algorithm);
        var reference = (BitmapSource)Get(shadow, "CurrentImageBitmap")!;
        Check(PixelHash(reference) != PixelHash(originalBitmap!), "原增强算法未实际执行，不能验证后台等价性");
        Check(PixelHash(reference) == PixelHash((BitmapSource)Get(vm, "CurrentImageBitmap")!), "后台增强像素与原算法不一致");
        Set(vm, "Brightness", 0.0);
        Set(vm, "Contrast", defaultContrast);
        Pump(250);
        Set(vm, "SearchText", "China");
        Set(vm, "SearchText", "India");
        Set(vm, "SearchText", "Japan");
        Pump(250);
        var visible = ((IEnumerable)Get(vm, "ImagesView")!).Cast<object>().ToArray();
        Check(visible.All(image => ((string)Get(image, "FilePath")!).Contains("Japan", StringComparison.OrdinalIgnoreCase)), "防抖筛选未采用最终搜索词");
        Set(vm, "SearchText", "");
        Pump(250);
        Check(((IEnumerable)Get(vm, "ImagesView")!).Cast<object>().Count() == images.Count, "清空筛选未恢复全集");

        var latencies = new List<double>();
        for (int i = 0; i < 30; i++)
        {
            Stopwatch timer = Stopwatch.StartNew();
            Set(vm, "CurrentImage", images[i % Math.Min(30, images.Count)]);
            timer.Stop();
            latencies.Add(timer.Elapsed.TotalMilliseconds);
            WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, "切图未完成");
        }
        object view = Create("Views.AnnotateView");
        ((FrameworkElement)view).DataContext = vm;
        var window = new Window { Content = view, Width = 1000, Height = 700, Left = -20000, Top = -20000, ShowInTaskbar = false };
        window.Show();
        Pump(300);
        int imageLists = 0;
        int realized = 0;
        Inspect(window);
        Set(vm, "IsGroupByFolder", true);
        Pump(250);
        Inspect(window);
        window.Close();
        latencies.Sort();
        string json = JsonSerializer.Serialize(new { mode = "ui", images = images.Count,
            navigation_median_ms = latencies[latencies.Count / 2], navigation_p95_ms = latencies[28],
            image_lists_checked = imageLists, realized_containers_max = realized,
            controlled_async_checks = optimized,
            checks = "rapid navigation, controlled cancellation, obsolete dispatcher commit, undo/redo during load, inherited annotation navigation, exact enhancement pixels, final search, WPF BAML, grouping" }, new JsonSerializerOptions { WriteIndented = true });
        if (args.Length > 3) File.WriteAllText(args[3], json);
        Console.WriteLine(json);
        return 0;

        void Inspect(DependencyObject node)
        {
            if (node is ListBox list && ReferenceEquals(list.ItemsSource, Get(vm, "ImagesView")))
            {
                imageLists++;
                realized = Math.Max(realized, list.Items.Cast<object>().Count(item => list.ItemContainerGenerator.ContainerFromItem(item) is not null));
            }
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(node); i++) Inspect(VisualTreeHelper.GetChild(node, i));
        }
    }

    private static string PixelHash(BitmapSource bitmap)
    {
        var normalized = new FormatConvertedBitmap(bitmap, PixelFormats.Bgra32, null, 0);
        int stride = normalized.PixelWidth * 4;
        byte[] pixels = new byte[stride * normalized.PixelHeight];
        normalized.CopyPixels(pixels, stride, 0);
        return Convert.ToHexString(SHA256.HashData(pixels));
    }

    private static void LoadAnnotations(object service, object image, object categories)
    {
        string path = (string)Get(image, "FilePath")!;
        string label = Path.ChangeExtension(path.Replace(Path.DirectorySeparatorChar + "images" + Path.DirectorySeparatorChar,
            Path.DirectorySeparatorChar + "labels" + Path.DirectorySeparatorChar), ".txt");
        if (File.Exists(label))
            service.GetType().GetMethod("LoadYoloAnnotations", All)!.Invoke(service, new[] { image, label, categories });
    }

    private static int Contracts(string[] args)
    {
        string destination = Path.GetFullPath(args[2]);
        Check(!Directory.Exists(destination), "任务契约验证目录已存在");
        Directory.CreateDirectory(destination);
        string source = Path.GetFullPath(args[3]);
        var reports = new List<object>();
        foreach (string task in new[] { "Detect", "Segment", "OBB", "Pose", "Classify" })
        {
            string root = Path.Combine(destination, task);
            Directory.CreateDirectory(Path.Combine(root, "images", "train"));
            File.WriteAllText(Path.Combine(root, ".novavision-performance-fixture"), "task-contract-test");
            string picture = Path.Combine(root, "images", "train", "sample.jpg");
            File.Copy(source, picture);
            object vm = Create("ViewModels.AnnotateViewModel");
            Set(vm, "AutoSaveEnabled", false);
            Set(vm, "CurrentDatasetPath", root);
            PropertyInfo taskProperty = vm.GetType().GetProperty("CurrentYoloTaskType", All)!;
            Set(vm, "CurrentYoloTaskType", Enum.Parse(taskProperty.PropertyType, task));
            IList categories = (IList)Get(vm, "Categories")!;
            categories.Clear();
            object category = Create("Models.CategoryItem");
            Set(category, "Id", 0); Set(category, "Name", "sample");
            categories.Add(category);
            object image = Create("Models.ImageItem");
            Set(image, "FilePath", picture); Set(image, "RelativePath", "train/sample.jpg");
            IList images = (IList)Get(vm, "Images")!;
            images.Add(image);
            Set(vm, "CurrentImage", image);
            WaitFor(() => Get(vm, "CurrentImageBitmap") is BitmapSource, task + " 图片未加载");
            if (task != "Classify")
            {
                object box = Create("Models.BoundingBox");
                Set(box, "ClassId", 0); Set(box, "ClassName", "sample");
                Set(box, "X", 0.2); Set(box, "Y", 0.2); Set(box, "W", 0.3); Set(box, "H", 0.3);
                Type shapeType = box.GetType().GetProperty("ShapeType", All)!.PropertyType;
                Set(box, "ShapeType", Enum.Parse(shapeType, task == "Segment" ? "Polygon" : task == "OBB" ? "OBB" : "Rectangle"));
                if (task is "Segment" or "OBB")
                    Set(box, "Points", new List<Point> { new(0.1, 0.1), new(0.4, 0.1), new(0.4, 0.4), new(0.1, 0.4) });
                if (task == "Pose")
                {
                    object keypoint = Create("Models.KeypointItem");
                    Set(keypoint, "Index", 0); Set(keypoint, "X", 0.25); Set(keypoint, "Y", 0.25); Set(keypoint, "Visibility", 2);
                    ((IList)Get(box, "Keypoints")!).Add(keypoint);
                }
                ((IList)Get(image, "BoundingBoxes")!).Add(box);
            }
            Call(vm, "SaveCurrentAnnotations");
            Call(vm, "UpdateCategoryCounts");
            VerifyCounts(vm);
            if (task != "Detect" && vm.GetType().GetMethod("__PerfOriginal_SaveCurrentAnnotations", All) is not null)
            {
                Type optimizer = AssemblyLoadContext.Default.Assemblies.Single(a => a.GetName().Name == "NovaVision.Performance").GetType("NovaVision.Performance.RuntimeOptimizer", true)!;
                Check((bool)optimizer.GetMethod("ShouldSyncMetadata")!.Invoke(null, new[] { vm })!, "非检测任务误用了元数据缓存");
            }
            var artifacts = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                .Where(path => Path.GetExtension(path) is ".txt" or ".yaml")
                .OrderBy(path => path, StringComparer.Ordinal)
                .ToDictionary(path => Path.GetRelativePath(root, path).Replace('\\', '/'),
                    path => File.ReadAllText(path).Replace(root.Replace('\\', '/'), "<dataset>").Replace(root, "<dataset>"));
            reports.Add(new { task, artifacts });
        }
        string json = JsonSerializer.Serialize(reports, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(destination, "contracts.json"), json);
        Console.WriteLine(json);
        return 0;
    }
}
