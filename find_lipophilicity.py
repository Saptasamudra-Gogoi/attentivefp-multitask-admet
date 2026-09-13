from docx import Document

doc = Document('manuscript_v3.docx')

for i, para in enumerate(doc.paragraphs):
    if 'lipophilicity' in para.text.lower() or 'logp' in para.text.lower():
        print(f"[Para {i}]: {para.text}\n")