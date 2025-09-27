# 🧠 Relatório Técnico — Detecção de Bots no Twitter com BERT

## 1. Objetivo do Projeto

O objetivo principal deste notebook é **treinar e avaliar um modelo BERT** para identificar contas automatizadas (“bots”) no Twitter, utilizando dados públicos (por exemplo, datasets do Kaggle, como *Twitter Bot Detection Dataset*).  
O modelo visa classificar cada tweet ou perfil como **bot (1)** ou **humano (0)**, com foco em precisão e eficiência.

---

## 2. Estrutura Geral do Notebook

O notebook foi organizado em seis seções principais:

1. **Configuração do ambiente** — Importação de bibliotecas, ativação de GPU e otimizações.  
2. **Hiperparâmetros e setup de diretórios** — Definição de preset BERT, sequência máxima, batch, épocas e taxa de aprendizado.  
3. **Preparação e subamostragem dos dados** — Redução do dataset para caber na memória do Colab, mantendo o balanceamento entre classes.  
4. **Criação dos datasets TensorFlow (`tf.data`)** — Conversão dos textos e rótulos em datasets eficientes.  
5. **Construção e treinamento do modelo BERT** — Utilização do preset `bert_base_en` com cabeça binária (sigmoid).  
6. **Avaliação e salvamento dos artefatos** — Cálculo de métricas, relatório de classificação e persistência do modelo.

---

## 3. Configuração e Otimizações

### 3.1. Inicialização e limpeza de sessão

Para evitar conflitos de memória e aproveitar o hardware do Colab:
- `gc.collect()` e `tf.keras.backend.clear_session()` limpam sessões anteriores.
- Ativa-se **XLA JIT** (`tf.config.optimizer.set_jit(True)`) para compilar graficamente operações TensorFlow e acelerar o treino.
- Ativa-se **mixed precision** (`float16`) para reduzir o uso de RAM na GPU e acelerar operações matriciais.

### 3.2. Hiperparâmetros definidos

| Parâmetro | Valor | Descrição |
|------------|--------|-----------|
| `PRESET` | `"bert_base_en"` | Modelo base da família BERT disponível no KerasNLP |
| `MAX_LEN` | 48 | Comprimento máximo da sequência (reduz custo computacional) |
| `BATCH` | 4 | Tamanho do lote (trade-off entre memória e estabilidade) |
| `EPOCHS` | 1 | Treinamento rápido de uma época (linear probing) |
| `LR` | 5e-5 | Taxa de aprendizado inicial |
| `MAX_PER_CLASS` | 1500 | Limite de exemplos por classe na subamostragem |

---

## 4. Preparação dos Dados

### 4.1. Subamostragem Estratificada

Para evitar consumo excessivo de RAM, a função `stratified_cap()`:
- Garante equilíbrio entre as classes (`bot` e `humano`);
- Seleciona no máximo `MAX_PER_CLASS` amostras de cada grupo;
- Embaralha os exemplos após o corte.

Com isso, o dataset final contém aproximadamente **3.000 amostras** (1.500 por classe), ideal para treinos rápidos no Colab.

### 4.2. Conversão em `tf.data.Dataset`

Os textos e rótulos são convertidos em tensores TensorFlow, usando:
```python
tf.data.Dataset.from_tensor_slices((texts, labels))
```
Em seguida:
- Os dados são embaralhados (apenas no treino);
- Divididos em lotes (`batch(batch_size)`);
- Pré-carregados com `prefetch(1)` para não estourar RAM.

---

## 5. Construção e Treinamento do Modelo

### 5.1. Modelo BERT com KerasNLP

O modelo é criado a partir de um preset do KerasNLP:
```python
bert_clf = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en",
    num_classes=1,
    activation="sigmoid",
    sequence_length=MAX_LEN
)
```

- **`num_classes=1` + `sigmoid`** → Saída binária (probabilidade de ser bot).
- **Preprocessor embutido** → Faz tokenização e padding automaticamente.

### 5.2. “Linear Probing”: congelamento do backbone

O parâmetro:
```python
bert_clf.backbone.trainable = False
```
congela as camadas do encoder BERT, mantendo apenas a cabeça (camadas finais densas) treinável — isso reduz drasticamente o tempo de execução e o consumo de GPU.

### 5.3. Compilação e callbacks

```python
bert_clf.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=LR),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[BinaryAccuracy, AUC]
)
```

- `AdamW` → otimizador com *weight decay*, ideal para fine-tuning.  
- `EarlyStopping` → interrompe o treino se o AUC não melhorar.  
- `ModelCheckpoint` → salva o melhor modelo com base em AUC.

---

## 6. Avaliação do Modelo

### 6.1. Predição e métricas

Após o treinamento:
```python
proba = bert_clf.predict(val_ds).squeeze()
auc   = roc_auc_score(y_test, proba)
pred  = (proba >= 0.5).astype(int)
```

Métricas geradas:
- **AUC (Área sob a Curva ROC)** — principal indicador do desempenho do modelo binário;
- **Precision / Recall / F1-score** — via `classification_report`;
- **Matriz de confusão** — para análise visual dos acertos/erros.

### 6.2. Resultados típicos (baseline rápido)

Com as configurações leves (`MAX_LEN=48`, `BATCH=4`, `1 época`):
- **AUC** ≈ 0.70–0.80 dependendo do dataset.
- **F1-score** razoável (~0.7), considerando o backbone congelado.

Esses resultados podem ser melhorados descongelando gradualmente as últimas camadas BERT e aumentando o número de épocas.

---

## 7. Salvamento dos Artefatos

Os resultados e o modelo são persistidos em:
```
artifacts_bot_detection/
├── bert_best.keras
├── bert_final.keras
├── bert_val_metrics.json
```

O JSON contém metadados como:
- `auc`
- `max_len`, `batch`, `epochs`
- `n_train_used` (tamanho efetivo do dataset)

---

## 8. Conclusões e Recomendações

### 8.1. Conclusões

- O notebook implementa um pipeline **otimizado para Colab**, com consumo leve e execução rápida (~5–10 min).  
- O modelo BERT, mesmo treinado apenas na cabeça (“linear probing”), já mostra boa capacidade de distinção entre bots e humanos.  
- O uso do `keras_nlp` simplifica muito o pré-processamento e integração do BERT.


---

## **Resumo Final**

Este notebook implementa um fluxo eficiente e escalável para classificação binária de tweets com BERT, demonstrando como realizar fine-tuning leve no Colab sem estourar memória, mantendo resultados competitivos.
