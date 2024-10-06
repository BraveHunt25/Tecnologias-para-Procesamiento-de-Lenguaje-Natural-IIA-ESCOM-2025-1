import os
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Cargar modelos
nlp_esp = spacy.load("es_core_news_sm")
nlp_eng = spacy.load("en_core_web_sm")

# Cargar archivos
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anexo_B_esp.txt'), 'r', encoding='utf-8') as f: 
    texto_esp = f.read()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anexo_B_eng.txt'), 'r', encoding='utf-8') as f: 
    texto_eng = f.read()

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
def remover_stop_words(tokens, nlp): 
    return [token for token in tokens if not nlp.vocab[token].is_stop]

def lematizar(tokens, nlp): 
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

def stemmer(word, idioma="es"):
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
def normalizar_texto(doc, idioma="es", nlp=nlp_esp):
    # Crear lista de tokens iniciales (eliminando puntuación y espacios)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    # 1. Eliminar palabras cortas
    tokens = eliminar_palabras_cortas(tokens)
    
    # 2. Eliminar palabras no alfabéticas
    tokens = eliminar_no_alfabeticos(tokens)
    
    # 3. Convertir palabras a minúsculas
    tokens = convertir_a_minusculas(tokens)
    
    # 4. Eliminar palabras stop
    tokens = remover_stop_words(tokens, nlp)
    
    # 5. Lematizar
    tokens = lematizar(tokens, nlp)
    
    # 6. Stemmizar
    tokens = [stemmer(token, idioma) for token in tokens]
    
    return tokens

# Aplicar la normalización
tokens_esp_normalizados = normalizar_texto(doc_esp, "es", nlp_esp)
tokens_eng_normalizados = normalizar_texto(doc_eng, "en", nlp_eng)

print(f"Texto normalizado en español: {tokens_esp_normalizados[:15]}")
print(f"Texto normalizado en inglés: {tokens_eng_normalizados[:15]}")

# Normalización iterativa
def normalizar_iterativamente(doc, idioma="es", iteraciones=1, nlp=None):
    # Primer ciclo de normalización
    tokens_normalizados = normalizar_texto(doc, idioma, nlp)
    
    for _ in range(iteraciones - 1):  # -1 porque ya se aplicó 1 iteración
        # Convertir la lista de tokens en un nuevo doc para la siguiente normalización
        nuevo_doc = nlp(" ".join(tokens_normalizados))
        
        # Aplicar la normalización de nuevo
        tokens_normalizados = normalizar_texto(nuevo_doc, idioma, nlp)
    
    return tokens_normalizados

tokens_esp_normalizados_iter = normalizar_iterativamente(doc_esp, "es", iteraciones=3, nlp=nlp_esp)
tokens_eng_normalizados_iter = normalizar_iterativamente(doc_eng, "en", iteraciones=3, nlp=nlp_eng)

print(f"Texto normalizado iterativamente en español: {tokens_esp_normalizados_iter[:15]}")
print(f"Texto normalizado iterativamente en inglés: {tokens_eng_normalizados_iter[:15]}")

with open("tokens_esp_normalizados.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(tokens_esp_normalizados_iter))
with open("tokens_eng_normalizados.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(tokens_eng_normalizados_iter))