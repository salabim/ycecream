import sys
import re

if len(sys.argv) == 1:
    print("specify a .py file to extract the version from")
    sys.exit()

pyfile_name = sys.argv[1]

if len(sys.argv)>=3:
    tomlfile_name = sys.argv[2]
else:
    tomlfile_name="pyproject.toml"

with open(pyfile_name, "r") as pyfile:
    s=pyfile.read()
    try:
        pre, rest=s.split('__version__ = "')
    except ValueError:
        print(f"no version info found in {pyfile_name}")
        sys.exit()
    version,post = rest.split('"',1)

with open(tomlfile_name, "r") as tomlfile:
    s=tomlfile.read()
    pre, rest=s.split('version = "')
    old_version,post = rest.split('"',1)

if old_version.startswith(version):
    if old_version==version:
        new_version_sub=0
    else:
        new_version_sub = int(old_version[len(version)+1:])+1
    version=f"{version}-{new_version_sub}"

s=f'{pre}version = "{version}"{post}'

with open(tomlfile_name,"w") as tomlfile:
    tomlfile.write(s)

print(f"put version {version} in {tomlfile_name}")
