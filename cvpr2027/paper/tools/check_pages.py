"""Report how many pages the main text uses (ICLR limit: 9). Reads build/main.aux after a build."""
import re, sys
aux = open("build/main.aux").read()
m = re.search(r"\\newlabel\{after-main-text\}\{\{[^}]*\}\{(\d+)\}", aux)
if not m:
    sys.exit("label after-main-text not found; build first")
pages = int(m.group(1)) - 1
print(f"main text: {pages} pages (limit 9){'  <-- OVER LIMIT' if pages > 9 else ''}")
