
import docx

def read_docx(path):
    doc = docx.Document(path)
    for i, para in enumerate(doc.paragraphs):
        if 0 <= i < 10:
            print(f"[{i}] {para.text}")

if __name__ == "__main__":
    read_docx(r"c:\Users\halfo\OneDrive\Documents\GitHub\whisperX-medical-transcribe\uploads\ГИ4_Тер_гос_Спб_КораблеваНВ_20.02.docx")
