
import os

from pdfrw import PdfReader, PdfWriter
from pdfrw.findobjs import page_per_xobj
import pdfrw
inpfn = 'helloworld.pdf'
outfn = 'extract.' + os.path.basename(inpfn)

pages = list(page_per_xobj(PdfReader(inpfn).pages, margin=0.5 * 72))
if not pages:
        raise IndexError("No XObjects found")
for page in pages:
        print(page)
        # print(page.get('/Type'))
# writer = PdfWriter(outfn)
# writer.addpages(pages)
# writer.write()
