import sys
import pymupdf
with pymupdf.open(sys.argv[1]) as document:
    page=document[int(sys.argv[2])]
    page.get_pixmap(matrix=pymupdf.Matrix(300/72,300/72),alpha=False).save(sys.argv[3])
