import os
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Cargar modelos
nlp_esp = spacy.load("es_core_news_sm")
nlp_eng = spacy.load("en_core_web_sm")

# Cargar archivos
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anexo_B_esp.txt'), 'r', encoding='utf-8') as f: texto_esp = f.read()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anexo_B_eng.txt'), 'r', encoding='utf-8') as f: texto_eng = f.read()

# Procesar textos
doc_esp = nlp_esp(texto_esp)
doc_eng = nlp_eng(texto_eng)

# Análisis inicial
total_tokens_esp = len(doc_esp)
total_tokens_eng = len(doc_eng)

print(f"Total de tokens en español: {total_tokens_esp}")
print(f"Total de tokens en inglés: {total_tokens_eng}")

unique_tokens_esp = len(set(token.text for token in doc_esp))
unique_tokens_eng = len(set(token.text for token in doc_eng))

print(f"Tokens únicos en español: {unique_tokens_esp}")
print(f"Tokens únicos en inglés: {unique_tokens_eng}")

# Contar y visualizar tokens más comunes
tokens_esp = [token.text.lower() for token in doc_esp if not token.is_punct and not token.is_space]
tokens_eng = [token.text.lower() for token in doc_eng if not token.is_punct and not token.is_space]

most_common_esp = Counter(tokens_esp).most_common(15)
most_common_eng = Counter(tokens_eng).most_common(15)

# Graficar tokens más comunes
def plot_most_common(tokens, title):
    tokens, counts = zip(*tokens)
    plt.figure(figsize=(10, 5))
    plt.bar(tokens, counts)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

plot_most_common(most_common_esp, "Tokens más comunes en español")
plot_most_common(most_common_eng, "Tokens más comunes en inglés")

# Funciones de normalización
def remover_stop_words(doc): 
    return [token.text for token in doc if not token.is_stop and not token.is_punct]

def lematizar(doc): 
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

def rudimentary_stemmer(word, idioma="es"):
    if idioma == "es": 
        suffixes = ['ar', 'er', 'ir', 'ando', 'iendo', 'ado', 'ido', 'ción', 'es', 'mente', 's', 'a', 'o']
    elif idioma == "en": 
        suffixes = ['ing', 'ed', 'ly', 's', 'es', 'er', 'est', 'y']
    for suffix in suffixes: 
        if word.endswith(suffix): 
            return word[:-len(suffix)]
    return word

def convertir_a_minusculas(tokens): 
    return [token.lower() for token in tokens]

def eliminar_no_alfabeticos(tokens): 
    return [token for token in tokens if token.isalpha()]

def eliminar_palabras_cortas(tokens, min_len=3): 
    return [token for token in tokens if len(token) >= min_len]

# Normalizar texto
def normalizar_texto(doc, idioma="es"):
    # Remover stop words
    tokens_sin_stop = remover_stop_words(doc)
    
    # Lematización
    tokens_lematizados = lematizar(doc)
    
    # Stemming rudimentario
    tokens_stemmizados = [rudimentary_stemmer(token, idioma) for token in tokens_lematizados]
    
    # Convertir a minúsculas
    tokens_minusculas = convertir_a_minusculas(tokens_stemmizados)
    
    # Eliminar caracteres no alfabéticos
    tokens_alfabeticos = eliminar_no_alfabeticos(tokens_minusculas)
    
    # Eliminar palabras cortas
    tokens_normalizados = eliminar_palabras_cortas(tokens_alfabeticos)    
    return tokens_normalizados

# Aplicar la normalización
tokens_esp_normalizados = normalizar_texto(doc_esp, "es")
tokens_eng_normalizados = normalizar_texto(doc_eng, "en")

print(f"Texto normalizado en español: {tokens_esp_normalizados[:15]}")
print(f"Texto normalizado en inglés: {tokens_eng_normalizados[:15]}")

# Normalización iterativa
# Aquí aplicamos el proceso de normalización al resultado obtenido
def normalizar_iterativamente(doc, idioma="es", iteraciones=1):
    tokens_normalizados = normalizar_texto(doc, idioma)
    for _ in range(iteraciones - 1):  # -1 porque ya hicimos 1 iteración
        # Convertir la lista de tokens en un nuevo doc para la siguiente normalización
        nuevo_doc = nlp_esp(" ".join(tokens_normalizados)) if idioma == "es" else nlp_eng(" ".join(tokens_normalizados))
        tokens_normalizados = normalizar_texto(nuevo_doc, idioma)
    return tokens_normalizados

tokens_esp_normalizados_iter = normalizar_iterativamente(doc_esp, "es", iteraciones=3)
tokens_eng_normalizados_iter = normalizar_iterativamente(doc_eng, "en", iteraciones=3)

print(f"Texto normalizado iterativamente en español: {tokens_esp_normalizados_iter[:15]}")
print(f"Texto normalizado iterativamente en inglés: {tokens_eng_normalizados_iter[:15]}")
