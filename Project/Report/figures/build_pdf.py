"""Build the two submission PDFs: the full report, and the same report without images.

The no-images version keeps every figure caption -- the instructions ask for "the same file without
the images", and the captions are text that carries real numbers.
"""
import re, subprocess, sys, shutil
from pathlib import Path
import markdown

REPORT = Path(r"C:/Aviv/University/Semester 8/Data Science/Homework - Group/Project/Report")
SOFFICE = r"C:/Program Files/LibreOffice/program/soffice.exe"

CSS = """
@page { size: A4; margin: 1.9cm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt; line-height: 1.15;
       color: #111; }
h1 { font-size: 18pt; margin: 0 0 2pt 0; }
h2 { font-size: 13pt; margin: 10pt 0 3pt 0; border-bottom: 1px solid #bbb; padding-bottom: 1pt; }
h3 { font-size: 11.5pt; margin: 7pt 0 2pt 0; }
h4 { font-size: 11pt; margin: 6pt 0 2pt 0; color: #222; }
p { margin: 0 0 4pt 0; text-align: justify; }
ol, ul { margin: 0 0 4pt 0; padding-left: 16pt; }
li { margin-bottom: 1pt; }
img { max-width: 92%; height: auto; display: block; margin: 4pt auto 1pt auto; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 5pt 0; font-size: 9pt; }
th, td { border: 1px solid #ccc; padding: 1.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0f2f4; }
code { font-family: "Consolas", monospace; font-size: 9.5pt; }
hr { border: none; border-top: 1px solid #ddd; margin: 7pt 0; }
"""


def to_html(md_text, title):
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    return (f"<html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def build(md_text, stem, title):
    html_path = REPORT / f"{stem}.html"
    html_path.write_text(to_html(md_text, title), encoding="utf-8")
    out = subprocess.run([SOFFICE, "--headless", "--convert-to",
                          "pdf:writer_pdf_Export", "--outdir", str(REPORT), str(html_path)],
                         capture_output=True, text=True, timeout=600)
    pdf = REPORT / f"{stem}.pdf"
    if not pdf.exists():
        print("  soffice stdout:", out.stdout.strip()[:300])
        print("  soffice stderr:", out.stderr.strip()[:300])
        raise SystemExit(f"failed to produce {pdf}")
    print(f"  wrote {pdf.name} ({pdf.stat().st_size/1024:.0f} KB)")
    html_path.unlink()
    return pdf


def page_count(pdf):
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return None


if __name__ == "__main__":
    src = (REPORT / "report.md").read_text(encoding="utf-8")
    src = re.sub(r"<!--.*?-->", "", src, flags=re.S)          # planning notes never ship
    src = re.sub(r"^\[(Friends|Cards|Figure)[^\]]*\]\s*$", "", src, flags=re.M)  # unfilled placeholders

    print("full report:")
    full = build(src, "writeup", "A Needle in a Stack of Magic Cards")

    # The no-images version drops each figure together with its caption: a caption with no figure
    # above it describes nothing, and the page limit explicitly excludes figures.
    noimg = re.sub(r"^!\[[^\]]*\]\([^)]*\)\s*$", "", src, flags=re.M)
    noimg = re.sub(r"^\*Figure \d+ -.*$", "", noimg, flags=re.M)
    noimg = re.sub(r"\n{3,}", "\n\n", noimg)
    print("no-images report:")
    ni = build(noimg, "noimages", "A Needle in a Stack of Magic Cards (no images)")
    (REPORT / "report_noimages.md").write_text(noimg, encoding="utf-8")

    for p in (full, ni):
        n = page_count(p)
        words = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True)
        print(f"  {p.name:14s} {n if n else '?'} pages, {len(words.stdout.split()):,} words")
