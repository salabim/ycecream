rmdir /s /q dist
for /d %%i in (ycecream-*) do rmdir /s /q "%%i"
rmdir /s /q ycecream.egg-info
copy *.md ycecream
call python -m toml_versioner ycecream/ycecream.py
call python -m build
call twine upload dist/*
rmdir /s /q dist
for /d %%i in (ycecream_*) do rmdir /s /q "%%i"
rmdir /s /q ycecream.egg-info
del ycecream\*.md