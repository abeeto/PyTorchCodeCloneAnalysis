import types
from pathlib import Path

from mypy.build import build
from mypy.fscache import FileSystemCache
from mypy.modulefinder import BuildSource
from mypy.options import Options
from mypy.traverser import TraverserVisitor
from mypy.types import AnyType

opts = Options()

opts = opts.apply_changes(
    {
        # "allow_empty_bodies": False,
        "allow_redefinition": False,
        "allow_untyped_globals": False,
        "always_false": [],
        "always_true": [],
        "bazel": False,
        "build_type": 0,
        "cache_dir": ".mypy_cache",
        "cache_fine_grained": True,
        "cache_map": {},
        "check_untyped_defs": True,
        "color_output": True,
        # "config_file": "/home/mlevental/dev_projects/TorchLite/try2/mypy.cfg",
        "debug_cache": False,
        # "disable_bytearray_promotion": False,
        "disable_error_code": ["name-defined"],
        # "disable_memoryview_promotion": False,
        # "disable_recursive_aliases": False,
        "disallow_any_decorated": False,
        "disallow_any_explicit": False,
        "disallow_any_expr": False,
        "disallow_any_generics": True,
        "disallow_any_unimported": False,
        "disallow_incomplete_defs": True,
        "disallow_subclassing_any": True,
        "disallow_untyped_calls": True,
        "disallow_untyped_decorators": True,
        "disallow_untyped_defs": True,
        "dump_build_stats": False,
        "dump_deps": False,
        "dump_graph": False,
        "dump_inference_stats": False,
        "dump_type_stats": False,
        "enable_error_code": [],
        # "enable_incomplete_feature": [],
        "enable_incomplete_features": False,
        "enable_recursive_aliases": False,
        "error_summary": True,
        "exclude": [],
        "explicit_package_bases": False,
        "export_types": True,
        "fast_exit": False,
        "fast_module_lookup": False,
        # CRUCIAL for not deleting parse tree
        # triggers TypeState.reset_all_subtype_caches() in build._build
        "fine_grained_incremental": True,
        "follow_imports": "normal",
        "follow_imports_for_stubs": False,
        # "hide_error_codes": False,
        "ignore_errors": True,
        "ignore_missing_imports": False,
        "ignore_missing_imports_per_module": False,
        # "implicit_optional": False,
        "implicit_reexport": False,
        "incremental": False,
        "inspections": False,
        "install_types": False,
        "local_partial_types": False,
        "logical_deps": False,
        "many_errors_threshold": 200,
        "mypy_path": [],
        "mypyc": False,
        "namespace_packages": True,
        "no_silence_site_packages": False,
        "no_site_packages": False,
        "non_interactive": False,
        "package_root": [],
        "pdb": False,
        "per_module_options": {},
        "platform": "linux",
        # "plugins": ["/home/mlevental/dev_projects/TorchLite/try2/plugin.py"],
        "preserve_asts": False,
        "pretty": False,
        # "python_executable": "/home/mlevental/dev_projects/TorchLite/venv/bin/python",
        "python_version": (3, 10),
        "raise_exceptions": True,
        "report_dirs": {},
        "scripts_are_modules": True,
        "semantic_analysis_only": False,
        "show_absolute_path": False,
        "show_column_numbers": False,
        "show_error_context": False,
        "show_error_end": False,
        "show_traceback": False,
        "skip_cache_mtime_checks": False,
        "skip_version_check": False,
        "sqlite_cache": False,
        "strict_concatenate": True,
        "strict_equality": True,
        "strict_optional": True,
        "use_builtins_fixtures": False,
        "use_fine_grained_cache": True,
        "verbosity": 0,
        "warn_incomplete_stub": False,
        "warn_no_return": True,
        "warn_redundant_casts": True,
        "warn_return_any": True,
        "warn_unreachable": False,
        "warn_unused_configs": True,
        "warn_unused_ignores": True,
    }
)

HURR_DIR = Path(__file__).parent
sources = [
    BuildSource(
        path=f'{(HURR_DIR / "demo.py").resolve()}',
        module="demo",
        base_dir=f'{HURR_DIR.resolve()}',
    )
]

fscache = FileSystemCache()

res = build(sources, options=opts, fscache=fscache)
astt = res.files["demo"]


class ProxyMethodWrapper:
    """
    Wrapper object for a method to be called.
    """

    def __init__(self, obj, func, name):
        self.obj, self.func, self.name = obj, func, name
        assert obj is not None
        assert func is not None
        assert name is not None

    def __call__(self, *args, **kwargs):
        for arg in args:
            if arg not in res.types or isinstance(res.types[arg], AnyType):
                continue
            typ = res.types[arg]
            print(str(arg), str(typ))

        return self.func(*args, **kwargs)


class PrintTypesVisitor(TraverserVisitor):
    def __getattribute__(self, name):
        """
        Return a proxy wrapper object if this is a method call.
        """
        att = object.__getattribute__(self, name)
        if name.startswith("_"):
            return att
        elif type(att) is types.MethodType or type(att) is types.BuiltinFunctionType:
            return ProxyMethodWrapper(self, att, name)
        else:
            return att


v = PrintTypesVisitor()
astt.accept(v)
