import ftplib


print("signing in")
ftp = ftplib.FTP("ftp.salabim.org", "ruudvander", "rthvdh")
ftp.cwd(f"domains/salabim.org/public_html")
print("connected")

file = "changelog.html"
with open(file, "rb") as fh:
    ftp.storbinary(f"STOR ycecream/changelog.html", fh)

file = "readme.html"
with open(file, "rb") as fh:
    ftp.storbinary(f"STOR ycecream/readme.html", fh)
    ftp.storbinary(f"STOR ycecream/index.html", fh)

file = "readme.html"
with open(file, "rb") as fh:
    ftp.storbinary(f"STOR ycecream/index.html", fh)

ftp.quit()

print("done")
