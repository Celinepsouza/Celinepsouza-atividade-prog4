# Relatório — Treinamento de BERT para Detecção de Bots no Twitter

Este relatório resume o notebook fornecido, destacando as etapas, decisões e objetivos.


# Twitter Bot Detection — Baseline + BERT (KerasNLP) — Colab

Notebook completo para rodar no **Google Colab** com:
- Baseline: TF-IDF + Logistic Regression
- **Fine-tuning do BERT (KerasNLP)**

> Dica: Se o seu CSV já estiver no Drive, use a seção de montagem do Drive. Ou faça upload direto.


## 0. Ambiente (Colab)

```python
# Se estiver no Colab, descomente para checar GPU:
```

```python
!pip install -U tensorflow tensorflow_text keras-nlp pandas scikit-learn
```

## 1. Dados: carregar CSV

```python
import os, pandas as pd, numpy as np
```

## 2. Colunas de texto e rótulo (com possibilidade de ajuste manual)

```python
def infer_cols(df):
```

```python
# Preparação básica
```

## 3. Split de treino/teste

```python
from sklearn.model_selection import train_test_split
```

## 4. Baseline: TF-IDF + Logistic Regression

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```

### 4.1 Salvar baseline

```python
from pathlib import Path, PurePosixPath
```

## 5. Fine-tuning BERT (KerasNLP)

```python
# ===== BERT ultra-rápido (KerasNLP, Colab) =====
```

### 5.1 Avaliação do BERT

```python
# Avaliação direta (métricas Keras)
```

```python
# Métricas detalhadas com sklearn (classification report, AUC, confusão)
```

```python
import matplotlib.pyplot as plt
```

### 5.2 Salvar modelo BERT e métricas

```python
import json
```


## 6. Notas
- Se a coluna inferida de rótulo não for **bot vs humano**, ajuste `label_col` manualmente na célula indicada.
- Para PT-BR ou multilíngue, considere presets multilingues (ex.: `bert_multi_cased` em libs alternativas) ou modelos do **Hugging Face Transformers**.
- A métrica **AUC** é recomendada para comparações; também avalie **F1** e **PR AUC** em datasets desbalanceados.

