using System.Security.Cryptography;
using System.Text.Json;
using dnlib.DotNet;
using dnlib.DotNet.Emit;
using dnlib.DotNet.Writer;

internal static class Patcher
{
    private const string ExpectedSha256 = "191A01EA651A721CEA708151A4F44617B6FE8C01F31D23BE2E82E5F2A207BE7F";
    private const string HelperTypeName = "NovaVision.Performance.RuntimeOptimizer";
    private const string VmTypeName = "AnnotateViewModel";
    private const string ViewTypeName = "AnnotateView";

    private static readonly string[] GuardedNames =
    [
        "UpdateCategoryCounts", "SaveCurrentAnnotations", "ApplyImageFilter",
        "ApplyImageEnhancement", "z"
    ];

    private static readonly string[] GetterNames =
    [
        "get_AnnotatedCountDisplay", "get_ImageListHeaderDisplay"
    ];

    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 2 && args[0] == "inspect")
            {
                Inspect(args[1]);
                return 0;
            }

            if (args.Length == 4 && args[0] == "patch")
            {
                Patch(args[1], args[2], args[3]);
                return 0;
            }

            if (args.Length == 4 && args[0] == "verify")
            {
                Verify(args[1], args[2], args[3]);
                return 0;
            }

            Console.Error.WriteLine("用法: Patcher inspect <NovaVision.dll> | patch <原DLL> <NovaVision.Performance.dll> <候选DLL> | verify <原DLL> <候选DLL> <报告.json>");
            return 2;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.ToString());
            return 1;
        }
    }

    private static void Inspect(string path)
    {
        using var module = ModuleDefMD.Load(path);
        var inspectMethods = new HashSet<string>(StringComparer.Ordinal)
        {
            "UpdateCategoryCounts", "get_AnnotatedCountDisplay", "get_ImageListHeaderDisplay",
            "SaveCurrentAnnotations", "ApplyImageFilter", "ApplyImageEnhancement", "z", ".ctor",
            "RefreshLicenseStatus", "SaveYoloAnnotations", "ToYoloLine"
        };
        var types = module.GetTypes()
            .Where(t => t.Name == VmTypeName || t.Name == ViewTypeName || t.Name == "MainViewModel"
                || t.Name == "YoloDatasetService" || t.Name == "BoundingBox")
            .Select(t => new
            {
                Type = t.FullName,
                Methods = t.Methods.Where(m => inspectMethods.Contains(m.Name.String)).Select(m => new
                {
                    Name = m.Name.String,
                    Token = m.MDToken.ToInt32().ToString("X8"),
                    Signature = m.MethodSig?.ToString(),
                    HasBody = m.HasBody,
                    InitLocals = m.HasBody ? m.Body.InitLocals : false,
                    MaxStack = m.HasBody ? m.Body.MaxStack : 0,
                    Instructions = m.HasBody ? m.Body.Instructions.Select((i, n) => new
                    {
                        Index = n,
                        Offset = i.Offset.ToString("X4"),
                        Op = i.OpCode.Name,
                        Operand = FormatOperand(i.Operand)
                    }).ToArray() : []
                }).ToArray()
            }).ToArray();

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            Path = Path.GetFullPath(path),
            Sha256 = HashFile(path),
            module.Assembly?.FullName,
            Types = types,
            ResourceNames = module.Resources.Select(r => r.Name.String).ToArray()
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static string FormatOperand(object? operand) => operand switch
    {
        null => "",
        Instruction instruction => $"IL_{instruction.Offset:X4}",
        Instruction[] targets => string.Join(",", targets.Select(t => $"IL_{t.Offset:X4}")),
        IMethod method => method.FullName,
        IField field => field.FullName,
        Local local => $"V_{local.Index}:{local.Type}",
        Parameter parameter => $"A_{parameter.Index}:{parameter.Type}",
        byte[] bytes => Convert.ToHexString(bytes),
        _ => operand.ToString() ?? ""
    };

    private static void Patch(string originalPath, string helperPath, string outputPath)
    {
        EnsureDifferentPaths(originalPath, outputPath);
        RequireExpectedSource(originalPath);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);

        using var module = ModuleDefMD.Load(originalPath);
        using var helperModule = ModuleDefMD.Load(helperPath);
        var before = CreateSnapshot(module);
        var helperType = helperModule.Find(HelperTypeName, false)
            ?? throw new InvalidOperationException($"helper 缺少类型 {HelperTypeName}");
        var helper = ResolveHelpers(module, helperType);
        var vmType = FindUniqueType(module, VmTypeName);
        var viewType = FindUniqueType(module, ViewTypeName);

        var update = FindUniqueMethod(vmType, "UpdateCategoryCounts");
        var annotatedGetter = FindUniqueMethod(vmType, "get_AnnotatedCountDisplay");
        var headerGetter = FindUniqueMethod(vmType, "get_ImageListHeaderDisplay");
        var save = FindUniqueMethod(vmType, "SaveCurrentAnnotations");
        var filter = FindUniqueMethod(vmType, "ApplyImageFilter");
        var enhancement = FindUniqueMethod(vmType, "ApplyImageEnhancement");
        var load = FindUniqueMethod(vmType, "z");
        var constructors = viewType.Methods.Where(m => m.IsConstructor && m.Name == ".ctor" && m.HasBody).ToArray();
        if (constructors.Length == 0)
            throw new InvalidOperationException($"{viewType.FullName} 没有可注入的实例构造函数");

        var changed = new HashSet<string>(StringComparer.Ordinal);
        CloneMethod(update, "__PerfOriginal_UpdateCategoryCounts");
        CloneMethod(save, "__PerfOriginal_SaveCurrentAnnotations");
        CloneMethod(filter, "__PerfOriginal_ApplyImageFilter");
        CloneMethod(enhancement, "__PerfOriginal_ApplyImageEnhancement");
        CloneMethod(load, "__PerfOriginal_z");
        changed.Add(MethodId(update));
        changed.Add(MethodId(annotatedGetter));
        changed.Add(MethodId(headerGetter));
        changed.Add(MethodId(save));
        changed.Add(MethodId(filter));
        changed.Add(MethodId(enhancement));
        changed.Add(MethodId(load));
        foreach (var ctor in constructors)
            changed.Add(MethodId(ctor));

        InjectVoidGuard(update, helper["TryUpdateCategoryCounts"]);
        InjectStringGetter(annotatedGetter, helper["AnnotatedCountDisplay"]);
        InjectStringGetter(headerGetter, helper["ImageListHeaderDisplay"]);
        InjectBeforeReturns(save, helper["AfterSaveAnnotations"]);
        InjectVoidGuard(save, helper["TrySaveCurrentAnnotations"]);
        InjectVoidGuard(filter, helper["TryApplyImageFilter"]);
        InjectVoidGuard(enhancement, helper["TryApplyImageEnhancement"]);
        InjectVoidGuard(load, helper["TryLoadCurrentImage"]);
        foreach (var ctor in constructors)
            InjectBeforeReturns(ctor, helper["AttachView"]);

        var metadataCalls = GuardCalls(save, "GenerateDataYaml", helper["ShouldSyncMetadata"]);
        var classCalls = GuardWriteAllLines(save, helper["ShouldSyncClasses"]);
        if (metadataCalls == 0 || classCalls == 0)
            throw new InvalidOperationException($"SaveCurrentAnnotations 目标点不足：GenerateDataYaml={metadataCalls}, File.WriteAllLines={classCalls}。未生成候选。");

        var lastSyncCall = save.Body.Instructions
            .Where(i => i.OpCode == OpCodes.Call || i.OpCode == OpCodes.Callvirt)
            .Where(i => i.Operand is IMethod m && (m.Name.String.Contains("GenerateDataYaml", StringComparison.Ordinal)
                || IsFileWriteAllLines(m)))
            .LastOrDefault() ?? throw new InvalidOperationException("未定位到 metadata/classes 写入调用");
        var syncIndex = save.Body.Instructions.IndexOf(lastSyncCall) + 1;
        save.Body.Instructions.Insert(syncIndex, Instruction.Create(OpCodes.Ldarg_0));
        save.Body.Instructions.Insert(syncIndex + 1, Instruction.Create(OpCodes.Call, helper["AfterSyncMetadata"]));

        var writeOptions = new ModuleWriterOptions(module);
        writeOptions.MetadataOptions.Flags |= MetadataFlags.PreserveRids | MetadataFlags.PreserveStringsOffsets
            | MetadataFlags.PreserveUSOffsets | MetadataFlags.KeepOldMaxStack;
        foreach (var bodyMethod in module.GetTypes().SelectMany(t => t.Methods).Where(m => m.HasBody))
            bodyMethod.Body.KeepOldMaxStack = true;
        foreach (var id in changed)
        {
            var changedMethod = module.GetTypes().SelectMany(t => t.Methods).Single(m => MethodId(m) == id);
            changedMethod.Body.MaxStack = (ushort)Math.Max((int)changedMethod.Body.MaxStack, 32);
        }
        module.Write(outputPath, writeOptions);

        using var written = ModuleDefMD.Load(outputPath);
        Audit(before, written, changed);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            Original = Path.GetFullPath(originalPath),
            Candidate = Path.GetFullPath(outputPath),
            OriginalSha256 = HashFile(originalPath),
            CandidateSha256 = HashFile(outputPath),
            ChangedMethods = changed.Order(StringComparer.Ordinal).ToArray(),
            AddedClones = new[]
            {
                MethodId(update) + " -> __PerfOriginal_UpdateCategoryCounts",
                MethodId(save) + " -> __PerfOriginal_SaveCurrentAnnotations",
                MethodId(filter) + " -> __PerfOriginal_ApplyImageFilter",
                MethodId(enhancement) + " -> __PerfOriginal_ApplyImageEnhancement",
                MethodId(load) + " -> __PerfOriginal_z"
            },
            MetadataGuardCallSites = metadataCalls,
            ClassesGuardCallSites = classCalls,
            ResourceCount = written.Resources.Count
        }, new JsonSerializerOptions { WriteIndented = true }));

        MethodDef CloneMethod(MethodDef source, string cloneName)
        {
            if (!source.HasBody)
                throw new InvalidOperationException($"不能克隆无方法体方法 {source.FullName}");
            if (source.DeclaringType.Methods.Any(m => m.Name == cloneName))
                throw new InvalidOperationException($"克隆方法已存在: {cloneName}");

            var clone = new MethodDefUser(cloneName, source.MethodSig,
                source.ImplAttributes, MethodAttributes.Private | MethodAttributes.HideBySig);
            foreach (var parameter in source.ParamDefs)
                clone.ParamDefs.Add(new ParamDefUser(parameter.Name, parameter.Sequence, parameter.Attributes));
            source.DeclaringType.Methods.Add(clone);
            clone.Body = CloneBody(source, clone);
            return clone;
        }
    }

    private static Dictionary<string, IMethod> ResolveHelpers(ModuleDef target, TypeDef helperType)
    {
        var names = new[]
        {
            "TryUpdateCategoryCounts", "AnnotatedCountDisplay", "ImageListHeaderDisplay",
            "TrySaveCurrentAnnotations", "TryApplyImageFilter", "TryApplyImageEnhancement",
            "TryLoadCurrentImage", "AttachView", "ShouldSyncMetadata", "ShouldSyncClasses",
            "AfterSyncMetadata", "AfterSaveAnnotations"
        };
        var result = new Dictionary<string, IMethod>(StringComparer.Ordinal);
        var importer = new Importer(target);
        foreach (var name in names)
        {
            var matches = helperType.Methods.Where(m => m.Name == name && m.IsStatic).ToArray();
            if (matches.Length != 1)
                throw new InvalidOperationException($"helper {name} 需要唯一静态方法，找到 {matches.Length} 个");
            var method = matches[0];
            if (method.MethodSig is null || method.MethodSig.Params.Count != 1
                || method.MethodSig.Params[0].ElementType != ElementType.Object)
                throw new InvalidOperationException($"helper {name} 必须接收一个 object 参数: {method.FullName}");
            var expectedReturn = name switch
            {
                "AnnotatedCountDisplay" or "ImageListHeaderDisplay" => ElementType.String,
                "AttachView" or "AfterSyncMetadata" or "AfterSaveAnnotations" => ElementType.Void,
                _ => ElementType.Boolean
            };
            if (method.MethodSig.RetType.ElementType != expectedReturn)
                throw new InvalidOperationException($"helper {name} 返回类型应为 {expectedReturn}: {method.FullName}");
            result.Add(name, importer.Import(method));
        }
        return result;
    }

    private static CilBody CloneBody(MethodDef source, MethodDef clone)
    {
        var sourceBody = source.Body;
        var body = new CilBody
        {
            InitLocals = sourceBody.InitLocals,
            MaxStack = sourceBody.MaxStack,
            KeepOldMaxStack = sourceBody.KeepOldMaxStack
        };
        var locals = new Dictionary<Local, Local>();
        foreach (var local in sourceBody.Variables)
        {
            var copy = new Local(local.Type) { Name = local.Name };
            body.Variables.Add(copy);
            locals.Add(local, copy);
        }
        var parameters = source.Parameters.ToDictionary(p => p.Index, p => clone.Parameters[p.Index]);
        var instructions = new Dictionary<Instruction, Instruction>();
        foreach (var instruction in sourceBody.Instructions)
        {
            var copy = Instruction.Create(OpCodes.Nop);
            copy.OpCode = instruction.OpCode;
            copy.Operand = instruction.Operand switch
            {
                Instruction => null,
                Instruction[] => null,
                Local local => locals[local],
                Parameter parameter => parameters[parameter.Index],
                _ => instruction.Operand
            };
            body.Instructions.Add(copy);
            instructions.Add(instruction, copy);
        }
        foreach (var instruction in sourceBody.Instructions)
        {
            var copy = instructions[instruction];
            copy.Operand = instruction.Operand switch
            {
                Instruction target => instructions[target],
                Instruction[] targets => targets.Select(t => instructions[t]).ToArray(),
                MethodDef method when method == source => clone,
                _ => copy.Operand
            };
        }
        foreach (var exception in sourceBody.ExceptionHandlers)
        {
            body.ExceptionHandlers.Add(new ExceptionHandler(exception.HandlerType)
            {
                CatchType = exception.CatchType,
                TryStart = Map(exception.TryStart),
                TryEnd = Map(exception.TryEnd),
                HandlerStart = Map(exception.HandlerStart),
                HandlerEnd = Map(exception.HandlerEnd),
                FilterStart = Map(exception.FilterStart)
            });
        }
        return body;

        Instruction? Map(Instruction? instruction) => instruction is null ? null : instructions[instruction];
    }

    private static void InjectVoidGuard(MethodDef method, IMethod helper)
    {
        RequireVoidInstanceMethod(method);
        var first = method.Body.Instructions[0];
        method.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Ldarg_0));
        method.Body.Instructions.Insert(1, Instruction.Create(OpCodes.Call, helper));
        method.Body.Instructions.Insert(2, Instruction.Create(OpCodes.Brfalse, first));
        method.Body.Instructions.Insert(3, Instruction.Create(OpCodes.Ret));
    }

    private static void InjectStringGetter(MethodDef method, IMethod helper)
    {
        if (!method.HasBody || method.IsStatic || method.MethodSig?.RetType.ElementType != ElementType.String)
            throw new InvalidOperationException($"getter 签名不符: {method.FullName}");
        var fallbackPop = Instruction.Create(OpCodes.Pop);
        method.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Ldarg_0));
        method.Body.Instructions.Insert(1, Instruction.Create(OpCodes.Call, helper));
        method.Body.Instructions.Insert(2, Instruction.Create(OpCodes.Dup));
        method.Body.Instructions.Insert(3, Instruction.Create(OpCodes.Brfalse, fallbackPop));
        method.Body.Instructions.Insert(4, Instruction.Create(OpCodes.Ret));
        method.Body.Instructions.Insert(5, fallbackPop);
    }

    private static void InjectBeforeReturns(MethodDef method, IMethod helper)
    {
        if (!method.HasBody || method.IsStatic || method.MethodSig?.RetType.ElementType != ElementType.Void)
            throw new InvalidOperationException($"构造函数签名不符: {method.FullName}");
        var returns = method.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToArray();
        if (returns.Length == 0)
            throw new InvalidOperationException($"构造函数无 ret: {method.FullName}");
        foreach (var ret in returns)
        {
            ret.OpCode = OpCodes.Ldarg_0;
            ret.Operand = null;
            var index = method.Body.Instructions.IndexOf(ret);
            method.Body.Instructions.Insert(index + 1, Instruction.Create(OpCodes.Call, helper));
            method.Body.Instructions.Insert(index + 2, Instruction.Create(OpCodes.Ret));
        }
    }

    private static int GuardCalls(MethodDef method, string nameFragment, IMethod guard)
    {
        var calls = method.Body.Instructions.Where(i => IsCall(i) && i.Operand is IMethod target
            && target.Name.String.Contains(nameFragment, StringComparison.Ordinal)).ToArray();
        foreach (var call in calls)
            InsertStackBalancedGuard(method, call, guard);
        return calls.Length;
    }

    private static int GuardWriteAllLines(MethodDef method, IMethod guard)
    {
        var calls = method.Body.Instructions.Where(i => IsCall(i) && i.Operand is IMethod target && IsFileWriteAllLines(target)).ToArray();
        foreach (var call in calls)
            InsertStackBalancedGuard(method, call, guard);
        return calls.Length;
    }

    private static bool IsFileWriteAllLines(IMethod method) => method.Name == "WriteAllLines"
        && method.DeclaringType.FullName == "System.IO.File"
        && method.MethodSig?.Params.Count >= 2;

    private static bool IsCall(Instruction instruction) => instruction.OpCode == OpCodes.Call || instruction.OpCode == OpCodes.Callvirt;

    private static void InsertStackBalancedGuard(MethodDef method, Instruction call, IMethod guard)
    {
        if (call.Operand is not IMethod target || target.MethodSig is null)
            throw new InvalidOperationException($"调用点缺少方法签名: {call}");
        var list = method.Body.Instructions;
        var index = list.IndexOf(call);
        if (index < 0 || index + 1 >= list.Count)
            throw new InvalidOperationException($"不能在方法尾部 guard 调用: {target.FullName}");
        var next = list[index + 1];
        var stackArgumentCount = target.MethodSig.Params.Count + (target.MethodSig.HasThis ? 1 : 0);
        if (stackArgumentCount == 0)
            throw new InvalidOperationException($"guard 目标没有参数栈: {target.FullName}");

        var branchToCall = Instruction.Create(OpCodes.Brtrue, call);
        list.Insert(index, Instruction.Create(OpCodes.Ldarg_0));
        list.Insert(index + 1, Instruction.Create(OpCodes.Call, guard));
        list.Insert(index + 2, branchToCall);
        var popIndex = index + 3;
        for (var i = 0; i < stackArgumentCount; i++)
            list.Insert(popIndex + i, Instruction.Create(OpCodes.Pop));
        var skipIndex = popIndex + stackArgumentCount;
        if (target.MethodSig.RetType.ElementType != ElementType.Void)
        {
            if (target.MethodSig.RetType.ElementType != ElementType.String
                && target.MethodSig.RetType.ElementType != ElementType.Object
                && target.MethodSig.RetType.ElementType != ElementType.Class
                && target.MethodSig.RetType.ElementType != ElementType.Array)
                throw new InvalidOperationException($"guard 暂不支持此返回类型的调用点: {target.FullName}");
            list.Insert(skipIndex++, Instruction.Create(OpCodes.Ldnull));
        }
        list.Insert(skipIndex, Instruction.Create(OpCodes.Br, next));
    }

    private static void Verify(string originalPath, string candidatePath, string reportPath)
    {
        RequireExpectedSource(originalPath);
        using var original = ModuleDefMD.Load(originalPath);
        using var candidate = ModuleDefMD.Load(candidatePath);
        var before = CreateSnapshot(original);
        var changed = InferChangedMethods(before, candidate);
        var allowed = AllowedMethods(before);
        Audit(before, candidate, allowed);
        var report = new
        {
            Status = "PASS",
            OriginalSha256 = HashFile(originalPath),
            CandidateSha256 = HashFile(candidatePath),
            OriginalMethodCount = before.Methods.Count,
            CandidateMethodCount = candidate.GetTypes().Sum(t => t.Methods.Count),
            ChangedMethods = changed.Order(StringComparer.Ordinal).ToArray(),
            ResourcesBytePreserved = true,
            ExistingMethodTokensPreserved = true
        };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(reportPath))!);
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine(JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static HashSet<string> InferChangedMethods(Snapshot before, ModuleDef candidate)
    {
        var changed = new HashSet<string>(StringComparer.Ordinal);
        foreach (var type in candidate.GetTypes())
        foreach (var method in type.Methods)
        {
            var id = MethodId(method);
            if (!before.Methods.TryGetValue(method.MDToken.ToInt32(), out var prior))
                continue;
            if (prior.Fingerprint != Fingerprint(method))
                changed.Add(id);
        }
        return changed;
    }

    private static HashSet<string> AllowedMethods(Snapshot before)
    {
        var names = new HashSet<string>(StringComparer.Ordinal)
        {
            "UpdateCategoryCounts", "get_AnnotatedCountDisplay", "get_ImageListHeaderDisplay",
            "SaveCurrentAnnotations", "ApplyImageFilter", "ApplyImageEnhancement", "z"
        };
        var allowed = before.Methods.Values.Where(m => names.Any(name => m.Id.EndsWith("::" + name, StringComparison.Ordinal))
                || m.Id.StartsWith("ViTooAI.Views.AnnotateView::.ctor", StringComparison.Ordinal))
            .Select(m => m.Id).ToHashSet(StringComparer.Ordinal);
        return allowed;
    }

    private sealed record MethodInfoSnapshot(string Id, string Signature, string Fingerprint, string Details, string BodyFingerprint);
    private sealed record Snapshot(Dictionary<int, MethodInfoSnapshot> Methods, Dictionary<string, string> Resources);

    private static Snapshot CreateSnapshot(ModuleDef module)
    {
        var methods = new Dictionary<int, MethodInfoSnapshot>();
        foreach (var method in module.GetTypes().SelectMany(t => t.Methods))
            methods.Add(method.MDToken.ToInt32(), new(MethodId(method), method.MethodSig?.ToString() ?? "", Fingerprint(method), FingerprintDetails(method), BodyFingerprint(method)));
        var resources = module.Resources.ToDictionary(r => r.Name.String, ResourceHash, StringComparer.Ordinal);
        return new Snapshot(methods, resources);
    }

    private static void Audit(Snapshot before, ModuleDef candidate, HashSet<string> allowedChanged)
    {
        var after = CreateSnapshot(candidate);
        foreach (var (token, prior) in before.Methods)
        {
            if (!after.Methods.TryGetValue(token, out var current))
                throw new InvalidOperationException($"现有方法 token 丢失: 0x{token:X8} {prior.Id}");
            if (prior.Id != current.Id || prior.Signature != current.Signature)
                throw new InvalidOperationException($"现有方法 token/signature 变化: 0x{token:X8} {prior.Id}");
            if (prior.Fingerprint != current.Fingerprint && !allowedChanged.Contains(prior.Id))
            {
                var method = candidate.GetTypes().SelectMany(t => t.Methods).Single(m => m.MDToken.ToInt32() == token);
                throw new InvalidOperationException($"非 allowlist 方法语义变化: {prior.Id}; original=[{prior.Details}]; current=[{current.Details}]; {DescribeMethod(method)}");
            }
        }

        var newMethods = after.Methods.Where(pair => !before.Methods.ContainsKey(pair.Key)).Select(pair => pair.Value.Id).ToArray();
        var expectedCloneNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "__PerfOriginal_UpdateCategoryCounts", "__PerfOriginal_SaveCurrentAnnotations",
            "__PerfOriginal_ApplyImageFilter", "__PerfOriginal_ApplyImageEnhancement", "__PerfOriginal_z"
        };
        var expectedType = before.Methods.Values.Select(m => m.Id).FirstOrDefault(id => id.EndsWith("::UpdateCategoryCounts", StringComparison.Ordinal))
            ?? throw new InvalidOperationException("allowlist 找不到 UpdateCategoryCounts 所在类型");
        var expectedNew = expectedCloneNames.Select(name => expectedType[..expectedType.LastIndexOf("::", StringComparison.Ordinal)] + "::" + name).ToHashSet(StringComparer.Ordinal);
        if (!newMethods.ToHashSet(StringComparer.Ordinal).SetEquals(expectedNew))
            throw new InvalidOperationException("新增方法不符合克隆 allowlist: " + string.Join(", ", newMethods));
        foreach (var (originalName, cloneName) in new (string Original, string Clone)[]
        {
            ("UpdateCategoryCounts", "__PerfOriginal_UpdateCategoryCounts"),
            ("SaveCurrentAnnotations", "__PerfOriginal_SaveCurrentAnnotations"),
            ("ApplyImageFilter", "__PerfOriginal_ApplyImageFilter"),
            ("ApplyImageEnhancement", "__PerfOriginal_ApplyImageEnhancement"),
            ("z", "__PerfOriginal_z")
        })
        {
            var originalInfo = before.Methods.Values.Single(m => m.Id == "ViTooAI.ViewModels.AnnotateViewModel::" + originalName);
            var clone = candidate.GetTypes().SelectMany(t => t.Methods).Single(m => m.Name == cloneName
                && m.DeclaringType.FullName == "ViTooAI.ViewModels.AnnotateViewModel");
            if (originalInfo.BodyFingerprint != BodyFingerprint(clone))
                throw new InvalidOperationException($"原方法克隆 IL 不一致: {originalName} / {cloneName}");
        }
        if (before.Resources.Count != after.Resources.Count || before.Resources.Any(r => !after.Resources.TryGetValue(r.Key, out var hash) || hash != r.Value))
            throw new InvalidOperationException("候选中的资源数量或资源字节发生变化");
    }

    private static string Fingerprint(MethodDef method)
    {
        return Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(FingerprintDetails(method))));
    }

    private static string BodyFingerprint(MethodDef method)
    {
        if (!method.HasBody)
            return "no-body";
        var lines = new List<string> { method.Body.InitLocals.ToString() };
        lines.AddRange(method.Body.Variables.Select(v => $"local:{v.Index}:{v.Type}"));
        lines.AddRange(method.Body.Instructions.Select(i => $"{CanonicalOpName(i.OpCode.Name)}|{NormalizeCloneOperand(method, i.Operand)}"));
        lines.AddRange(method.Body.ExceptionHandlers.Select(e => $"eh:{e.HandlerType}:{Offset(e.TryStart)}:{Offset(e.TryEnd)}:{Offset(e.HandlerStart)}:{Offset(e.HandlerEnd)}:{Offset(e.FilterStart)}:{e.CatchType}"));
        return Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(string.Join("\n", lines))));

        string Offset(Instruction? instruction) => instruction is null ? "-" : method.Body.Instructions.IndexOf(instruction).ToString();
    }

    private static string CanonicalOpName(string name) => name.EndsWith(".s", StringComparison.Ordinal)
        && name is "br.s" or "brfalse.s" or "brtrue.s" or "beq.s" or "bge.s" or "bgt.s" or "ble.s" or "blt.s"
            or "bne.un.s" or "bge.un.s" or "bgt.un.s" or "ble.un.s" or "blt.un.s" or "leave.s"
        ? name[..^2]
        : name;

    private static string NormalizeCloneOperand(MethodDef owner, object? operand) => operand switch
    {
        MethodDef method when method == owner => "method:<self>",
        Instruction instruction => "target:" + owner.Body.Instructions.IndexOf(instruction),
        Instruction[] targets => "targets:" + string.Join(",", targets.Select(t => owner.Body.Instructions.IndexOf(t))),
        _ => NormalizeOperand(owner, operand)
    };

    private static string FingerprintDetails(MethodDef method)
    {
        if (!method.HasBody)
            return $"{method.ImplAttributes}:{method.Attributes}:no-body";
        var variables = method.Body.Variables.ToArray();
        var initLocals = variables.Length == 0 ? "unused" : method.Body.InitLocals.ToString();
        var lines = new List<string> { $"{method.ImplAttributes}:{method.Attributes}:{initLocals}" };
        lines.AddRange(variables.Select(v => $"local:{v.Index}:{v.Type}"));
        lines.AddRange(method.Body.Instructions.Select(i => $"{i.OpCode.Name}|{NormalizeOperand(method, i.Operand)}"));
        lines.AddRange(method.Body.ExceptionHandlers.Select(e => $"eh:{e.HandlerType}:{Offset(e.TryStart)}:{Offset(e.TryEnd)}:{Offset(e.HandlerStart)}:{Offset(e.HandlerEnd)}:{Offset(e.FilterStart)}:{e.CatchType}"));
        return string.Join("\n", lines);

        string Offset(Instruction? instruction) => instruction is null ? "-" : method.Body.Instructions.IndexOf(instruction).ToString();
    }

    private static string DescribeMethod(MethodDef method) => $"impl={method.ImplAttributes}, attrs={method.Attributes}, body={(method.HasBody ? "yes" : "no")}, init={method.Body?.InitLocals}, vars={method.Body?.Variables.Count}, il={method.Body?.Instructions.Count}, eh={method.Body?.ExceptionHandlers.Count}, max={method.Body?.MaxStack}";

    private static string NormalizeOperand(MethodDef owner, object? operand) => operand switch
    {
        null => "",
        Instruction instruction => "target:" + owner.Body.Instructions.IndexOf(instruction),
        Instruction[] targets => "targets:" + string.Join(",", targets.Select(t => owner.Body.Instructions.IndexOf(t))),
        IMethod method => "method:" + method.FullName,
        IField field => "field:" + field.FullName,
        Local local => $"local:{local.Index}:{local.Type}",
        Parameter parameter => $"parameter:{parameter.Index}:{parameter.Type}",
        _ => operand.ToString() ?? ""
    };

    private static string ResourceHash(Resource resource)
    {
        if (resource is not EmbeddedResource embedded)
            return $"{resource.Attributes}:{resource.GetType().FullName}";
        return $"{resource.Attributes}:{Convert.ToHexString(SHA256.HashData(embedded.CreateReader().ToArray()))}";
    }

    private static TypeDef FindUniqueType(ModuleDef module, string simpleName)
    {
        var matches = module.GetTypes().Where(t => t.Name == simpleName).ToArray();
        if (matches.Length != 1)
            throw new InvalidOperationException($"类型 {simpleName} 需要唯一匹配，找到 {matches.Length} 个: {string.Join(", ", matches.Select(t => t.FullName))}");
        return matches[0];
    }

    private static MethodDef FindUniqueMethod(TypeDef type, string name)
    {
        var matches = type.Methods.Where(m => m.Name == name).ToArray();
        if (matches.Length != 1)
            throw new InvalidOperationException($"方法 {type.FullName}::{name} 需要唯一匹配，找到 {matches.Length} 个");
        return matches[0];
    }

    private static void RequireVoidInstanceMethod(MethodDef method)
    {
        if (!method.HasBody || method.IsStatic || method.MethodSig?.RetType.ElementType != ElementType.Void)
            throw new InvalidOperationException($"void guard 签名不符: {method.FullName}");
    }

    private static string MethodId(MethodDef method) => $"{method.DeclaringType.FullName}::{method.Name}";
    private static string HashFile(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));

    private static void RequireExpectedSource(string path)
    {
        var actual = HashFile(path);
        if (!string.Equals(actual, ExpectedSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"源 DLL SHA256 不匹配。期望 {ExpectedSha256}，实际 {actual}");
    }

    private static void EnsureDifferentPaths(string originalPath, string outputPath)
    {
        if (string.Equals(Path.GetFullPath(originalPath), Path.GetFullPath(outputPath), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("候选输出路径不能覆盖源 DLL");
    }
}
