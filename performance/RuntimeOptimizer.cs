using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Specialized;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;

namespace NovaVision.Performance;

/// <summary>
/// NovaVision 运行时辅助逻辑。通过缓存反射访问宿主对象，避免依赖宿主程序集编译类型。
/// </summary>
public static class RuntimeOptimizer
{
    private static readonly ConditionalWeakTable<object, RuntimeState> States = new();
    private static readonly ConcurrentDictionary<(Type Type, string Name), PropertyInfo> Properties = new();
    private static readonly PropertyInfo MissingProperty = typeof(string).GetProperty(nameof(string.Length))!;
    private static readonly ConcurrentDictionary<(Type Type, string Name), MethodInfo> Methods = new();
    private static readonly MethodInfo MissingMethod = typeof(object).GetMethod(nameof(ToString), Type.EmptyTypes)!;
    private static readonly ConcurrentDictionary<(Type Type, string Name), FieldInfo> Fields = new();
    private static readonly FieldInfo MissingField = typeof(RuntimeOptimizer).GetField(
        nameof(MissingField), BindingFlags.Static | BindingFlags.NonPublic)!;
    private static readonly string[] MetadataPropertyNames =
    [
        "CurrentDatasetPath",
        "CurrentYoloTaskType",
        "Categories",
        "Images"
    ];

    /// <summary>
    /// 使用图片和框集合的变更事件维护类别计数，并保留宿主的可观察属性与重入语义。
    /// </summary>
    public static bool TryUpdateCategoryCounts(object vm)
    {
        if (vm is null)
            return false;

        try
        {
            var state = States.GetValue(vm, static _ => new RuntimeState());
            EnsureInvalidationSubscriptions(vm, state);
            lock (state.Gate)
            {
                var busyField = GetField(vm.GetType(), "ch");
                if (busyField is null || busyField.GetValue(vm) is not bool busy || busy)
                    return false;

                busyField.SetValue(vm, true);
                try
                {
                    if (!state.StatisticsInitialized)
                        InitializeStatistics(vm, state);

                    ApplyCachedCategoryCounts(vm, state);
                    state.StatisticsInitialized = true;
                    return true;
                }
                finally
                {
                    busyField.SetValue(vm, false);
                }
            }
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// 返回 null 时由补丁器调用宿主原始显示 getter。
    /// </summary>
    public static string? AnnotatedCountDisplay(object vm)
    {
        if (vm is null || !States.TryGetValue(vm, out var state))
            return null;

        lock (state.Gate)
        {
            if (!state.StatisticsInitialized)
                return null;

            var images = GetPropertyValue(vm, "Images");
            if (!TryGetCount(images, out var imageCount))
                return null;

            return $"已标注: {state.AnnotatedImageCount} / {imageCount} 张图像";
        }
    }

    /// <summary>
    /// 返回 null 时由补丁器调用宿主原始显示 getter。
    /// </summary>
    public static string? ImageListHeaderDisplay(object vm)
    {
        if (vm is null)
            return null;

        try
        {
            var images = GetPropertyValue(vm, "Images");
            var view = GetPropertyValue(vm, "ImagesView");
            if (!TryGetCount(images, out var total) || !TryGetCount(view, out var visible))
                return null;

            if (total == 0)
                return "📁 图像列表";

            return visible == total
                ? $"📁 图像列表 ({total} 张)"
                : $"📁 图像列表 (筛选: {visible} / {total} 张)";
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// 仅在本 ViewModel 曾正常保存过同一张图、标签字节和元数据均与当前状态一致时跳过重写。
    /// </summary>
    public static bool TrySaveCurrentAnnotations(object vm)
    {
        if (vm is null || ShouldSyncMetadata(vm))
            return false;

        try
        {
            var state = States.GetValue(vm, static _ => new RuntimeState());
            var image = GetPropertyValue(vm, "CurrentImage");
            if (image is null || !state.StatisticsInitialized)
                return false;
            // 空且未标负的图片需要由原保存流程删除标签，不能与空负样本等同。
            if (GetPropertyValue(image, "IsNegativeSample") is not true
                && GetPropertyValue(image, "BoundingBoxes") is ICollection emptyBoxes && emptyBoxes.Count == 0)
                return false;

            var snapshot = BuildAnnotationSnapshot(vm, image);
            lock (state.Gate)
            {
                if (state.LastSavedAnnotation is not { } saved
                    || !ReferenceEquals(saved.Image, image)
                    || !StringComparer.OrdinalIgnoreCase.Equals(saved.Path, snapshot.Path)
                    || !StringComparer.Ordinal.Equals(saved.ContentHash, snapshot.ContentHash)
                    || !StringComparer.Ordinal.Equals(saved.MetadataSignature, BuildMetadataSignature(vm, state)))
                    return false;
            }

            if (!File.Exists(snapshot.Path) || !File.ReadAllBytes(snapshot.Path).AsSpan().SequenceEqual(snapshot.Content))
                return false;

            GetMethod(image.GetType(), "UpdateAnnotationStatus")?.Invoke(image, null);
            return TryUpdateCategoryCounts(vm);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// 由补丁器在旧保存方法正常返回后调用，验证吞异常的标签写入结果。
    /// </summary>
    public static void AfterSaveAnnotations(object vm)
    {
        if (vm is null)
            return;

        try
        {
            var state = States.GetValue(vm, static _ => new RuntimeState());
            EnsureInvalidationSubscriptions(vm, state);
            var image = GetPropertyValue(vm, "CurrentImage");
            if (image is null)
                return;

            var snapshot = BuildAnnotationSnapshot(vm, image);
            if (!File.Exists(snapshot.Path) || !File.ReadAllBytes(snapshot.Path).AsSpan().SequenceEqual(snapshot.Content))
            {
                lock (state.Gate)
                    state.LastSavedAnnotation = null;
                return;
            }

            var metadataSignature = BuildMetadataSignature(vm, state);
            lock (state.Gate)
            {
                state.LastSavedAnnotation = new SavedAnnotationSnapshot(
                    image, snapshot.Path, snapshot.ContentHash,
                    state.MetadataInvalidated ? null : state.SyncedMetadataSignature ?? metadataSignature);
            }
        }
        catch
        {
            if (States.TryGetValue(vm, out var state))
            {
                lock (state.Gate)
                    state.LastSavedAnnotation = null;
            }
        }
    }

    /// <summary>
    /// 将筛选变更交给负责界面调度与防抖的辅助逻辑。
    /// </summary>
    public static bool TryApplyImageFilter(object vm) => InteractionOptimizer.TryApplyImageFilter(vm);

    /// <summary>
    /// 将增强变更交给负责界面调度与防抖的辅助逻辑。
    /// </summary>
    public static bool TryApplyImageEnhancement(object vm) => InteractionOptimizer.TryApplyImageEnhancement(vm);

    /// <summary>
    /// 将当前图像的解码与显示交给有代际保护的辅助逻辑。
    /// </summary>
    public static bool TryLoadCurrentImage(object vm) => InteractionOptimizer.TryLoadCurrentImage(vm);

    /// <summary>
    /// 将宿主视图与 ViewModel 的可观察变更关联起来。
    /// </summary>
    public static void AttachView(object view)
    {
        if (view is null)
            return;

        try
        {
            InteractionOptimizer.AttachView(view);
            var dataContext = GetPropertyValue(view, "DataContext");
            if (dataContext is null)
                return;

            var state = States.GetValue(dataContext, static _ => new RuntimeState());
            lock (state.Gate)
            {
                state.AttachedView = view;
                EnsureInvalidationSubscriptions(dataContext, state);
            }
        }
        catch
        {
            // UI 生命周期中的反射失效不应阻断宿主视图加载。
        }
    }

    /// <summary>
    /// 决定是否需要再次生成数据集元数据；首次、未识别或签名变化时始终同步。
    /// </summary>
    public static bool ShouldSyncMetadata(object vm)
    {
        if (vm is null)
            return true;

        try
        {
            if (!CanCacheMetadataForTask(vm))
                return true;
            if (!CurrentImageBelongsToDataset(vm))
                return true;
            var state = States.GetValue(vm, static _ => new RuntimeState());
            EnsureInvalidationSubscriptions(vm, state);
            var signature = BuildMetadataSignature(vm, state);
            lock (state.Gate)
            {
                if (state.MetadataInvalidated || state.SyncedMetadataSignature is null)
                    return true;

                if (!StringComparer.Ordinal.Equals(signature, state.SyncedMetadataSignature))
                    return true;

                return !ArtifactsUnchanged(state.MetadataArtifacts);
            }
        }
        catch
        {
            return true;
        }
    }

    /// <summary>
    /// 决定是否需要再次写入 classes.txt；不能解析类别或目标文件时保留原写操作。
    /// </summary>
    public static bool ShouldSyncClasses(object vm)
    {
        if (vm is null)
            return true;

        try
        {
            if (!CanCacheMetadataForTask(vm))
                return true;
            if (!CurrentImageBelongsToDataset(vm))
                return true;
            var state = States.GetValue(vm, static _ => new RuntimeState());
            EnsureInvalidationSubscriptions(vm, state);
            var signature = BuildCategorySignature(vm);
            lock (state.Gate)
            {
                if (state.MetadataInvalidated || state.SyncedCategorySignature is null)
                    return true;

                if (!StringComparer.Ordinal.Equals(signature, state.SyncedCategorySignature))
                    return true;

                var classesPath = ClassesPath(vm);
                return classesPath is null
                    || !ArtifactUnchanged(classesPath, state.ClassesArtifact)
                    || !ClassesFileMatchesCategories(vm, classesPath);
            }
        }
        catch
        {
            return true;
        }
    }

    /// <summary>
    /// 仅由补丁器在元数据生成步骤正常完成后调用。
    /// </summary>
    public static void AfterSyncMetadata(object vm)
    {
        if (vm is null)
            return;

        try
        {
            if (!CanCacheMetadataForTask(vm))
                return;
            var state = States.GetValue(vm, static _ => new RuntimeState());
            EnsureInvalidationSubscriptions(vm, state);
            var metadataSignature = BuildMetadataSignature(vm, state);
            var categorySignature = BuildCategorySignature(vm);
            lock (state.Gate)
            {
                state.SyncedMetadataSignature = metadataSignature;
                state.SyncedCategorySignature = categorySignature;
                state.MetadataArtifacts = CaptureMetadataArtifacts(vm);
                if (state.MetadataArtifacts.Count < 2)
                {
                    state.SyncedMetadataSignature = null;
                    state.SyncedCategorySignature = null;
                    state.MetadataInvalidated = true;
                    return;
                }

                var classesPath = ClassesPath(vm);
                if (classesPath is null || !ClassesFileMatchesCategories(vm, classesPath))
                {
                    state.SyncedMetadataSignature = null;
                    state.SyncedCategorySignature = null;
                    state.MetadataInvalidated = true;
                    return;
                }

                state.ClassesArtifact = classesPath is null ? null : CaptureArtifact(classesPath);
                state.MetadataInvalidated = false;
            }
        }
        catch
        {
            // 状态未更新时，下次保存会再次执行元数据同步。
        }
    }

    private static void EnsureInvalidationSubscriptions(object vm, RuntimeState state)
    {
        if (vm is INotifyPropertyChanged vmNotifier && state.VmPropertyChangedHandler is null)
        {
            PropertyChangedEventHandler handler = (_, args) =>
            {
                if (args.PropertyName is null || MetadataPropertyNames.Contains(args.PropertyName, StringComparer.Ordinal))
                    InvalidateMetadata(state);
                if (args.PropertyName is null or "Images")
                {
                    lock (state.Gate)
                    {
                        var imagesCollection = state.ImagesCollection;
                        var imagesHandler = state.ImagesCollectionHandler;
                        SubscribeCollectionInvalidation(
                            GetPropertyValue(vm, "Images"), state,
                            ref imagesCollection, ref imagesHandler, vm);
                        state.ImagesCollection = imagesCollection;
                        state.ImagesCollectionHandler = imagesHandler;
                        if (state.StatisticsInitialized)
                            ReconcileImages(vm, state);
                    }
                }
            };
            vmNotifier.PropertyChanged += handler;
            state.VmPropertyChangedHandler = handler;
        }

        var images = GetPropertyValue(vm, "Images");
        var imagesCollection = state.ImagesCollection;
        var imagesHandler = state.ImagesCollectionHandler;
        SubscribeCollectionInvalidation(images, state, ref imagesCollection, ref imagesHandler, vm);
        state.ImagesCollection = imagesCollection;
        state.ImagesCollectionHandler = imagesHandler;
        var categories = GetPropertyValue(vm, "Categories");
        var categoriesCollection = state.CategoriesCollection;
        var categoriesHandler = state.CategoriesCollectionHandler;
        SubscribeCollectionInvalidation(categories, state, ref categoriesCollection, ref categoriesHandler, vm);
        state.CategoriesCollection = categoriesCollection;
        state.CategoriesCollectionHandler = categoriesHandler;
    }

    private static void SubscribeCollectionInvalidation(
        object? collection,
        RuntimeState state,
        ref object? savedCollection,
        ref NotifyCollectionChangedEventHandler? savedHandler,
        object vm)
    {
        if (ReferenceEquals(collection, savedCollection))
            return;

        if (savedCollection is INotifyCollectionChanged oldNotifier && savedHandler is not null)
            oldNotifier.CollectionChanged -= savedHandler;

        savedCollection = collection;
        savedHandler = null;
        if (collection is INotifyCollectionChanged notifier)
        {
            NotifyCollectionChangedEventHandler handler = (_, _) =>
            {
                InvalidateMetadata(state);
                if (ReferenceEquals(collection, state.ImagesCollection))
                {
                    lock (state.Gate)
                    {
                        if (state.StatisticsInitialized)
                            ReconcileImages(vm, state);
                    }
                }
            };
            notifier.CollectionChanged += handler;
            savedHandler = handler;
        }
    }

    private static void InvalidateMetadata(RuntimeState state)
    {
        lock (state.Gate)
        {
            state.MetadataInvalidated = true;
            state.MetadataRevision++;
            state.SyncedMetadataSignature = null;
            state.SyncedCategorySignature = null;
        }
    }

    private static void InitializeStatistics(object vm, RuntimeState state)
    {
        foreach (var image in state.ImageStates.Keys.ToArray())
            RemoveImageState(state, image);
        state.CategoryCounts.Clear();
        state.AnnotatedImageCount = 0;
        var images = GetPropertyValue(vm, "Images") as IEnumerable
            ?? throw new InvalidOperationException("Images 集合不可用");

        foreach (var image in images)
        {
            if (image is null)
                continue;
            RecountImage(vm, state, image);
        }

        state.StatisticsInitialized = true;
    }

    private static void ReconcileImages(object vm, RuntimeState state)
    {
        var images = GetPropertyValue(vm, "Images") as IEnumerable
            ?? throw new InvalidOperationException("Images 集合不可用");
        var current = new HashSet<object>(ReferenceEqualityComparer.Instance);
        foreach (var image in images)
        {
            if (image is null)
                continue;
            current.Add(image);
            if (!state.ImageStates.ContainsKey(image))
                RecountImage(vm, state, image);
        }

        foreach (var stale in state.ImageStates.Keys.Where(image => !current.Contains(image)).ToArray())
            RemoveImageState(state, stale);
    }

    private static void RecountImage(object vm, RuntimeState state, object image)
    {
        if (!state.ImageStates.TryGetValue(image, out var imageState))
        {
            imageState = new ImageState(image);
            state.ImageStates.Add(image, imageState);
            imageState.PropertyChangedHandler = (_, args) =>
            {
                lock (state.Gate)
                    RecountImage(vm, state, image);
                if (args.PropertyName is null or "FilePath" or "RelativePath")
                    InvalidateMetadata(state);
            };
            if (image is INotifyPropertyChanged imageNotifier)
                imageNotifier.PropertyChanged += imageState.PropertyChangedHandler;
        }
        else
        {
            ApplyContribution(state, imageState.Contribution, -1);
            UnsubscribeBoxes(imageState);
        }

        imageState.Contribution = ReadContribution(image);
        ApplyContribution(state, imageState.Contribution, 1);
        SubscribeBoxes(vm, state, imageState);
    }

    private static ImageContribution ReadContribution(object image)
    {
        var isAnnotated = GetPropertyValue(image, "IsAnnotated") is bool annotated && annotated;
        var classCounts = new Dictionary<int, int>();
        var boxes = GetPropertyValue(image, "BoundingBoxes") as IEnumerable
            ?? throw new InvalidOperationException("BoundingBoxes 集合不可用");
        foreach (var box in boxes)
        {
            if (box is null)
                continue;
            if (GetPropertyValue(box, "ClassId") is not int classId)
                throw new InvalidOperationException("BoundingBox.ClassId 不可用");
            classCounts[classId] = classCounts.GetValueOrDefault(classId) + 1;
        }

        return new ImageContribution(isAnnotated, classCounts);
    }

    private static void ApplyContribution(RuntimeState state, ImageContribution contribution, int sign)
    {
        if (contribution.IsAnnotated)
            state.AnnotatedImageCount += sign;

        foreach (var pair in contribution.ClassCounts)
        {
            var next = state.CategoryCounts.GetValueOrDefault(pair.Key) + sign * pair.Value;
            if (next <= 0)
                state.CategoryCounts.Remove(pair.Key);
            else
                state.CategoryCounts[pair.Key] = next;
        }
    }

    private static void SubscribeBoxes(object vm, RuntimeState state, ImageState imageState)
    {
        var collection = GetPropertyValue(imageState.Image, "BoundingBoxes");
        imageState.BoxCollection = collection;
        if (collection is INotifyCollectionChanged notifier)
        {
            NotifyCollectionChangedEventHandler handler = (_, _) =>
            {
                lock (state.Gate)
                    RecountImage(vm, state, imageState.Image);
            };
            notifier.CollectionChanged += handler;
            imageState.BoxCollectionChangedHandler = handler;
        }

        if (collection is IEnumerable boxes)
        {
            foreach (var box in boxes)
            {
                if (box is not INotifyPropertyChanged propertyNotifier)
                    continue;

                PropertyChangedEventHandler handler = (_, _) =>
                {
                    lock (state.Gate)
                        RecountImage(vm, state, imageState.Image);
                };
                propertyNotifier.PropertyChanged += handler;
                imageState.BoxPropertyChangedHandlers.Add((propertyNotifier, handler));
            }
        }
    }

    private static void UnsubscribeBoxes(ImageState imageState)
    {
        if (imageState.BoxCollection is INotifyCollectionChanged notifier && imageState.BoxCollectionChangedHandler is not null)
            notifier.CollectionChanged -= imageState.BoxCollectionChangedHandler;
        imageState.BoxCollection = null;
        imageState.BoxCollectionChangedHandler = null;
        foreach (var (propertyNotifier, handler) in imageState.BoxPropertyChangedHandlers)
            propertyNotifier.PropertyChanged -= handler;
        imageState.BoxPropertyChangedHandlers.Clear();
    }

    private static void RemoveImageState(RuntimeState state, object image)
    {
        if (!state.ImageStates.Remove(image, out var imageState))
            return;

        ApplyContribution(state, imageState.Contribution, -1);
        UnsubscribeBoxes(imageState);
        if (image is INotifyPropertyChanged notifier && imageState.PropertyChangedHandler is not null)
            notifier.PropertyChanged -= imageState.PropertyChangedHandler;
    }

    private static void ApplyCachedCategoryCounts(object vm, RuntimeState state)
    {
        var categories = GetPropertyValue(vm, "Categories") as IEnumerable
            ?? throw new InvalidOperationException("Categories 集合不可用");
        var selected = GetPropertyValue(vm, "SelectedCategory");
        var index = 0;
        foreach (var category in categories)
        {
            if (category is null || GetPropertyValue(category, "Id") is not int id)
                throw new InvalidOperationException("CategoryItem.Id 不可用");

            SetPropertyIfDifferent(category, "Count", state.CategoryCounts.GetValueOrDefault(id));
            SetProperty(category, "ShortcutIndex", index < 9 ? index + 1 : -1);
            SetProperty(category, "IsSelected", ReferenceEquals(category, selected));
            index++;
        }

        var notify = GetMethod(vm.GetType(), "NotifyImageCountsChanged")
            ?? throw new MissingMethodException(vm.GetType().FullName, "NotifyImageCountsChanged");
        notify.Invoke(vm, null);
    }

    private static void SetPropertyIfDifferent(object target, string name, object value)
    {
        var property = GetProperty(target.GetType(), name)
            ?? throw new MissingMemberException(target.GetType().FullName, name);
        if (Equals(property.GetValue(target), value))
            return;
        property.SetValue(target, value);
    }

    private static void SetProperty(object target, string name, object value)
    {
        var property = GetProperty(target.GetType(), name)
            ?? throw new MissingMemberException(target.GetType().FullName, name);
        property.SetValue(target, value);
    }

    private static string BuildMetadataSignature(object vm, RuntimeState state)
    {
        var categorySignature = BuildCategorySignature(vm);
        var path = GetPropertyValue(vm, "CurrentDatasetPath") as string;
        var task = GetPropertyValue(vm, "CurrentYoloTaskType");
        var images = GetPropertyValue(vm, "Images");
        var imageIdentity = images is null ? "null" : RuntimeHelpers.GetHashCode(images).ToString(System.Globalization.CultureInfo.InvariantCulture);
        var canonical = string.Join("\n", categorySignature, path ?? "?", task?.ToString() ?? "?", imageIdentity,
            state.MetadataRevision.ToString(System.Globalization.CultureInfo.InvariantCulture));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(canonical)));
    }

    private static string BuildCategorySignature(object vm)
    {
        if (GetProperty(vm.GetType(), "Categories") is null)
            throw new InvalidOperationException("Categories 属性不可用");

        var categories = GetPropertyValue(vm, "Categories") as IEnumerable;
        if (categories is null)
            throw new InvalidOperationException("Categories 集合不可用");

        var items = new List<string>();
        foreach (var category in categories)
        {
            if (category is null)
                throw new InvalidOperationException("类别项为空");

            var id = GetPropertyValue(category, "Id")?.ToString();
            var name = GetPropertyValue(category, "Name") as string;
            if (id is null || name is null)
                throw new InvalidOperationException("类别标识或名称不可用");

            items.Add(id + "\t" + name);
        }

        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join("\n", items))));
    }

    private static IReadOnlyDictionary<string, ArtifactSnapshot> CaptureMetadataArtifacts(object vm)
    {
        var artifacts = new Dictionary<string, ArtifactSnapshot>(StringComparer.OrdinalIgnoreCase);
        foreach (var fileName in new[] { "data.yaml", "classes.txt" })
        {
            var datasetPath = GetPropertyValue(vm, "CurrentDatasetPath") as string;
            if (string.IsNullOrWhiteSpace(datasetPath))
                continue;
            var path = Path.Combine(datasetPath, fileName);
            if (File.Exists(path))
                artifacts[path] = CaptureArtifact(path);
        }

        return artifacts;
    }

    private static bool ArtifactsUnchanged(IReadOnlyDictionary<string, ArtifactSnapshot>? artifacts)
    {
        if (artifacts is null || artifacts.Count == 0)
            return false;

        foreach (var entry in artifacts)
        {
            if (!ArtifactUnchanged(entry.Key, entry.Value))
                return false;
        }

        return true;
    }

    private static bool ArtifactUnchanged(string path, ArtifactSnapshot? expected)
    {
        if (expected is null)
            return false;

        var current = CaptureArtifact(path);
        return current.Length == expected.Length
            && current.LastWriteUtc == expected.LastWriteUtc
            && StringComparer.Ordinal.Equals(current.Sha256, expected.Sha256);
    }

    private static ArtifactSnapshot CaptureArtifact(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists)
            throw new FileNotFoundException("元数据文件不存在", path);

        using var stream = File.OpenRead(path);
        var hash = Convert.ToHexString(SHA256.HashData(stream));
        return new ArtifactSnapshot(info.Length, info.LastWriteTimeUtc, hash);
    }

    private static string? ClassesPath(object vm)
    {
        var datasetPath = GetPropertyValue(vm, "CurrentDatasetPath") as string;
        return string.IsNullOrWhiteSpace(datasetPath)
            ? null
            : Path.Combine(datasetPath, "classes.txt");
    }

    private static bool CurrentImageBelongsToDataset(object vm)
    {
        try
        {
            var datasetPath = GetPropertyValue(vm, "CurrentDatasetPath") as string;
            var currentImage = GetPropertyValue(vm, "CurrentImage");
            var imagePath = currentImage is null ? null : GetPropertyValue(currentImage, "FilePath") as string;
            if (string.IsNullOrWhiteSpace(datasetPath) || string.IsNullOrWhiteSpace(imagePath))
                return false;

            var root = Path.GetFullPath(datasetPath);
            var path = Path.IsPathRooted(imagePath)
                ? Path.GetFullPath(imagePath)
                : Path.GetFullPath(Path.Combine(root, imagePath));
            var relative = Path.GetRelativePath(root, path);
            var first = relative.Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries)
                .FirstOrDefault();
            return StringComparer.OrdinalIgnoreCase.Equals(first, "images");
        }
        catch
        {
            return false;
        }
    }

    private static bool CanCacheMetadataForTask(object vm)
    {
        var task = GetPropertyValue(vm, "CurrentYoloTaskType")?.ToString();
        return StringComparer.OrdinalIgnoreCase.Equals(task, "Detect");
    }

    private static bool ClassesFileMatchesCategories(object vm, string path)
    {
        try
        {
            if (!File.Exists(path))
                return false;

            var categories = GetPropertyValue(vm, "Categories") as IEnumerable;
            if (categories is null)
                return false;

            var expected = new List<(int Id, string Name)>();
            foreach (var category in categories)
            {
                if (category is null || GetPropertyValue(category, "Id") is not int id
                    || GetPropertyValue(category, "Name") is not string name)
                    return false;
                expected.Add((id, name));
            }

            var expectedLines = expected.OrderBy(item => item.Id).Select(item => item.Name).ToArray();
            var actualLines = File.ReadAllLines(path);
            return expectedLines.SequenceEqual(actualLines, StringComparer.Ordinal);
        }
        catch
        {
            return false;
        }
    }

    private static AnnotationSnapshot BuildAnnotationSnapshot(object vm, object image)
    {
        var datasetPath = GetPropertyValue(vm, "CurrentDatasetPath") as string;
        var filePath = GetPropertyValue(image, "FilePath") as string;
        if (string.IsNullOrWhiteSpace(datasetPath) || string.IsNullOrWhiteSpace(filePath))
            throw new InvalidOperationException("当前数据集路径或图片路径不可用");

        var root = Path.GetFullPath(datasetPath);
        var imagePath = Path.IsPathRooted(filePath)
            ? Path.GetFullPath(filePath)
            : Path.GetFullPath(Path.Combine(root, filePath));
        var relativeImagePath = Path.GetRelativePath(root, imagePath);
        var segments = relativeImagePath.Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 2 || !StringComparer.OrdinalIgnoreCase.Equals(segments[0], "images"))
        {
            var relative = GetPropertyValue(image, "RelativePath") as string;
            if (string.IsNullOrWhiteSpace(relative))
                throw new InvalidOperationException("图片相对路径不可用");
            segments = relative.Split([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar], StringSplitOptions.RemoveEmptyEntries);
            if (segments.Length > 0 && StringComparer.OrdinalIgnoreCase.Equals(segments[0], "images"))
                segments = segments.Skip(1).ToArray();
        }
        else
        {
            segments = segments.Skip(1).ToArray();
        }

        var relativeLabelPath = Path.ChangeExtension(Path.Combine(segments), ".txt");
        var labelsDirectory = Path.GetFullPath(Path.Combine(root, "labels"));
        var labelPath = Path.GetFullPath(Path.Combine(labelsDirectory, relativeLabelPath));
        if (!labelPath.StartsWith(labelsDirectory + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("标签路径超出数据集 labels 目录");

        var lines = new StringBuilder();
        var boxes = GetPropertyValue(image, "BoundingBoxes") as IEnumerable
            ?? throw new InvalidOperationException("BoundingBoxes 集合不可用");
        foreach (var box in boxes)
        {
            if (box is null)
                continue;
            var toYoloLine = GetMethod(box.GetType(), "ToYoloLine")
                ?? throw new MissingMethodException(box.GetType().FullName, "ToYoloLine");
            if (toYoloLine.Invoke(box, null) is not string line)
                throw new InvalidOperationException("ToYoloLine 未返回字符串");
            lines.AppendLine(line);
        }

        var content = new UTF8Encoding(false).GetBytes(lines.ToString());
        var statusPrefix = GetPropertyValue(image, "IsNegativeSample") is true ? "negative:" : "normal:";
        var contentHash = statusPrefix + Convert.ToHexString(SHA256.HashData(content));
        return new AnnotationSnapshot(labelPath, content, contentHash);
    }

    private static bool TryGetCount(object? collection, out int count)
    {
        if (collection is ICollection genericCollection)
        {
            count = genericCollection.Count;
            return true;
        }

        if (collection is not null)
        {
            var countValue = GetPropertyValue(collection, "Count");
            if (countValue is int value)
            {
                count = value;
                return true;
            }
        }

        count = 0;
        return false;
    }

    private static object? GetPropertyValue(object target, string name)
    {
        var property = GetProperty(target.GetType(), name);
        return property?.GetValue(target);
    }

    private static PropertyInfo? GetProperty(Type type, string name)
    {
        var property = Properties.GetOrAdd(
            (type, name),
            static key => key.Type.GetProperty(key.Name, BindingFlags.Instance | BindingFlags.Public) ?? MissingProperty);
        return ReferenceEquals(property, MissingProperty) ? null : property;
    }

    private static MethodInfo? GetMethod(Type type, string name)
    {
        var method = Methods.GetOrAdd(
            (type, name),
            static key => key.Type.GetMethod(key.Name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic) ?? MissingMethod);
        return ReferenceEquals(method, MissingMethod) ? null : method;
    }

    private static FieldInfo? GetField(Type type, string name)
    {
        var field = Fields.GetOrAdd(
            (type, name),
            static key => key.Type.GetField(key.Name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic) ?? MissingField);
        return ReferenceEquals(field, MissingField) ? null : field;
    }

    private sealed class RuntimeState
    {
        public object Gate { get; } = new();
        public object? AttachedView { get; set; }
        public object? ImagesCollection { get; set; }
        public object? CategoriesCollection { get; set; }
        public bool MetadataInvalidated { get; set; } = true;
        public long MetadataRevision { get; set; }
        public string? SyncedMetadataSignature { get; set; }
        public string? SyncedCategorySignature { get; set; }
        public IReadOnlyDictionary<string, ArtifactSnapshot>? MetadataArtifacts { get; set; }
        public ArtifactSnapshot? ClassesArtifact { get; set; }
        public PropertyChangedEventHandler? VmPropertyChangedHandler { get; set; }
        public NotifyCollectionChangedEventHandler? ImagesCollectionHandler { get; set; }
        public NotifyCollectionChangedEventHandler? CategoriesCollectionHandler { get; set; }
        public SavedAnnotationSnapshot? LastSavedAnnotation { get; set; }
        public bool StatisticsInitialized { get; set; }
        public int AnnotatedImageCount { get; set; }
        public Dictionary<int, int> CategoryCounts { get; } = new();
        public Dictionary<object, ImageState> ImageStates { get; } = new(ReferenceEqualityComparer.Instance);
    }

    private sealed class ImageState(object image)
    {
        public object Image { get; } = image;
        public ImageContribution Contribution { get; set; } = new(false, new Dictionary<int, int>());
        public PropertyChangedEventHandler? PropertyChangedHandler { get; set; }
        public object? BoxCollection { get; set; }
        public NotifyCollectionChangedEventHandler? BoxCollectionChangedHandler { get; set; }
        public List<(INotifyPropertyChanged Notifier, PropertyChangedEventHandler Handler)> BoxPropertyChangedHandlers { get; } = new();
    }

    private sealed record ImageContribution(bool IsAnnotated, Dictionary<int, int> ClassCounts);
    private sealed record AnnotationSnapshot(string Path, byte[] Content, string ContentHash);
    private sealed record SavedAnnotationSnapshot(object Image, string Path, string ContentHash, string? MetadataSignature);
    private sealed record ArtifactSnapshot(long Length, DateTime LastWriteUtc, string Sha256);
}
