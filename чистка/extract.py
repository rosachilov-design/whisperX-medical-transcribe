import docx

doc = docx.Document('uploads/interview3.docx')
with open('temp.txt', 'w', encoding='utf-8') as f:
    for p in doc.paragraphs:
        f.write(p.text + '\n')
