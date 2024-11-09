import pdfplumber                           # Para la lectura y archivos PDF
import nltk                                 # Para métodos de procesamiento de lenguaje
import os                                   # Para resolver rutas de descargas de modelos
from nltk.tokenize import word_tokenize     # Método para tokenizar
nltk.download('punkt_tab')                      # Modelo para tokenizar
nltk.download('averaged_perceptron_tagger_eng') # Modelo para el POS-Tagging
ruta_doc_1 = './documentos/documento-1.pdf'
with pdfplumber.open(ruta_doc_1) as pdf:
    # Extraer texto de todas las páginas
    doc_1 = ""                        # Inicializamos con una cadena vacía el documento
    for page in pdf.pages:            # Por cada página en las páginas del pdf...
        doc_1 += page.extract_text()  # agregamos el texto a lo que se extraiga en la página

    # Imprimimos el texto resultante
    print(doc_1)
# Ruta del documento PDF 2
ruta_doc_2 = './documentos/documento-2.pdf'
# Abrir el PDF
with pdfplumber.open(ruta_doc_2) as pdf:
    # Extraer texto de todas las páginas
    doc_2 = ""                          # Inicializamos con una cadena vacía el documento
    for page in pdf.pages:              # Por cada página en las páginas del pdf...
        doc_2 += page.extract_text()    # agregamos el texto a lo que se extraiga en la página

    # Imprimimos el texto resultante
    print(doc_2)
# Ruta del documento PDF 3
ruta_doc_3 = './documentos/documento-3.pdf'

# Abrir el PDF
with pdfplumber.open(ruta_doc_3) as pdf:
    # Extraer texto de todas las páginas
    doc_3 = ""                          # Inicializamos con una cadena vacía el documento
    for page in pdf.pages:              # Por cada página en las páginas del pdf...
        doc_3 += page.extract_text()    # agregamos el texto a lo que se extraiga en la página

    # Imprimimos el texto resultante
    print(doc_3)

doc_1 = doc_1.lower()
doc_2 = doc_2.lower()
doc_3 = doc_3.lower()
#Imprimimos resultados
print(doc_1)
print(doc_2)
print(doc_3)
# Tokenizamos los documentos, cabe mencionar que el tipo de dato acá cambia de cadena a lista
doc_1 = word_tokenize(doc_1, "english")
doc_2 = word_tokenize(doc_2, "english")
doc_3 = word_tokenize(doc_3, "english")
doc_1 = nltk.pos_tag(doc_1)
doc_2 = nltk.pos_tag(doc_2)
doc_3 = nltk.pos_tag(doc_3)
print(doc_1)
print(doc_2)
print(doc_3)



