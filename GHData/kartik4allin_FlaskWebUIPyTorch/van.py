import pyvan 

OPTIONS = {"main_file_name": "fileuploadmain.py", 
            "show_console": False,
            "use_pipreqs": True,
            "install_only_these_modules": [],
            "exclude_modules":[],
            "include_modules":[],
            "path_to_get_pip_and_python_embeded_zip": ""
            }

pyvan.build(OPTIONS)