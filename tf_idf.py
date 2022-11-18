# -*- coding: utf-8 -*-
# Enzo Bloss Stival
"""Tarefa 6 - TFIDF e Cosseno
1. Sua tarefa será gerar a matriz termo-documento usando TF-IDF por meio da aplicação das  fórmulas  TF-IDF  na  matriz  termo-documento  criada  com  a  utilização  do  algoritmo  
Bag of Words. Sobre o Corpus que recuperamos anteriormente. O entregável desta tarefa é uma matriz termo-documento onde a primeira linha são os termos e as linhas subsequentes são 
os vetores calculados com o TF-IDF. 
2. Sua tarefa será gerar uma matriz de distância, computando o cosseno do ângulo entre todos  os vetores que encontramos usando o tf-idf. Para isso use a seguinte fórmula para o 
cálculo do  cosseno  use  a  fórmula  apresentada  em  Word2Vector  (frankalcantara.com)  (https://frankalcantara.com/Aulas/Nlp/out/Aula4.html#/0/4/2)  e  apresentada  na  figura  
a seguir: O resultado deste trabalho será uma matriz que relaciona cada um dos vetores já calculados com todos os outros vetores disponíveis na matriz termo-documento mostrando a 
distância entre cada um destes vetores. 
"""
from bs4 import BeautifulSoup
import requests
import string
import numpy 
import re
import pandas
import xarray

sentencas = []
urls = ["https://www.ibm.com/cloud/learn/natural-language-processing", 
"https://www.techtarget.com/searchenterpriseai/definition/natural-language-processing-NLP", 
"https://www.datarobot.com/blog/what-is-natural-language-processing-introduction-to-nlp/",
 "https://hbr.org/2022/04/the-power-of-natural-language-processing", 
 "https://machinelearningmastery.com/natural-language-processing/"]
for link in urls:
    url = requests.get(link).content
    html = BeautifulSoup(url, "html.parser")
    for data in html(['style', 'script']):
        data.decompose()
    html = ' '.join(html.stripped_strings)
    html = re.sub("[\n\t]", "", html)
    html = re.split("[.!:?;]", html)
    sentencas.append(html)
palavras = set()
matriz = []
for sentenca in sentencas:
    for frase in sentenca:
        for palavra in frase.split():
            palavras.add(palavra)
palavras = list(palavras)
contagem = [0] * len(palavras)
#print(palavras)

for sentenca in sentencas:
    for frase in sentenca:
        for palavra in frase.split():
            contagem[palavras.index(palavra)] += 1
    matriz.append(contagem)
    contagem = [0] * len(palavras)

newMatriz = matriz
newMatriz2 = matriz

newMatriz = pandas.DataFrame(matriz, columns=palavras)
newMatriz2 = pandas.DataFrame(matriz, columns=palavras)


newMatriz =  newMatriz / len(palavras)
newMatriz2 =  numpy.log10(newMatriz2 / len(palavras))

TFIDF = newMatriz * newMatriz2
cossine=numpy.dot(newMatriz.to_xarray(),newMatriz2.to_xarray())/(numpy.linalg.norm(newMatriz.to_xarray()) * numpy.linalg.norm(newMatriz2.to_xarray()))
display(newMatriz)
display(newMatriz2)
display(TFIDF)

display(cossine)