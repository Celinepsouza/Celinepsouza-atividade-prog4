# 🧠 Relatório Técnico — Twitter Bot Detection com BERT (KerasNLP)

## 1. Objetivo do Projeto

O objetivo deste projeto foi desenvolver e avaliar um **modelo baseado em BERT** para detectar **bots no Twitter**.  
A proposta consiste em treinar uma rede neural pré-treinada (`bert_base_en`) para classificar contas como **humanas (0)** ou **automatizadas (1)**, a partir de textos e metadados contidos no dataset “Twitter Bot Detection”, disponível no Kaggle.

O foco principal foi **reduzir o tempo de execução** e **otimizar o uso de recursos** para que o modelo pudesse ser treinado rapidamente no **Google Colab**, sem comprometer o desempenho.

---

## 2. Estrutura Geral do Notebook

O notebook foi estruturado em cinco partes principais:

1. **Configuração e otimização do ambiente**  
   - Limpeza de sessão (`gc.collect()` e `clear_session()`);
   - Ativação de **XLA (Accelerated Linear Algebra)** para acelerar o cálculo vetorial;
   - Ativação de **mixed precision (`float16`)** para reduzir o uso de memória e aumentar a velocidade na GPU.

2. **Definição dos hiperparâmetros principais**  
   Incluindo preset do BERT, tamanho de sequência, batch size, número de épocas e taxa de aprendizado.

3. **Pré-processamento e subamostragem dos dados**  
   - Uso de função `stratified_cap()` para limitar o número de exemplos por classe e manter equilíbrio entre bots e humanos;
   - Redução do dataset para poucas milhares de amostras, ideal para Colab.

4. **Construção e treinamento do modelo BERT**  
   - Utilização do `keras_nlp.models.BertClassifier.from_preset("bert_base_en")`;
   - Backbone do BERT **congelado** (estratégia *linear probing*);
   - Treinamento leve (1 época), com otimização via `AdamW`.

5. **Avaliação e salvamento do modelo**  
   - Cálculo de AUC, acurácia e matriz de confusão;
   - Exportação do modelo em formato `.keras` e/ou `.h5`.

---

## 3. Configuração e Otimizações

Para garantir um treinamento rápido e eficiente, o notebook adota várias estratégias:

- **XLA JIT Compilation** (`tf.config.optimizer.set_jit(True)`): compila operações do TensorFlow para código otimizado.
- **Mixed Precision** (`set_global_policy("mixed_float16")`): reduz o tamanho dos tensores para `float16`, aproveitando o hardware da GPU.
- **Batch pequeno (4 ou 2)**: evita *Out of Memory* no Colab.
- **MAX_LEN reduzido (24–48 tokens)**: menor sequência → menor custo computacional.
- **1 única época (EPOCHS=1)**: suficiente para validar a viabilidade do modelo.
- **Subamostragem estratificada**: seleciona até 800–1500 exemplos por classe, equilibrando o dataset.

Essas configurações tornam o notebook **executável em menos de 10 minutos**, mesmo em ambientes gratuitos do Colab.

---

## 4. Pipeline de Dados

O pipeline utiliza o `tf.data.Dataset` para criar conjuntos eficientes:

```python
tf.data.Dataset.from_tensor_slices((texts, labels))
```

- **Embaralhamento** apenas no treino (`shuffle(buffer_size=min(8000, len(texts)))`);
- **Batching** dinâmico (`batch(batch_size, drop_remainder=False)`);
- **Prefetch(1)**: garante pipeline estável e economiza RAM;
- **Execução não determinística (`deterministic=False`)**: libera otimizações automáticas de I/O no TensorFlow.

---

## 5. Construção e Treinamento do Modelo

### 5.1. Arquitetura BERT

O modelo é inicializado com:

```python
bert_clf = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en",
    num_classes=1,
    activation="sigmoid",
    sequence_length=MAX_LEN
)
```

- **`num_classes=1` + `sigmoid`** → saída binária (probabilidade de ser bot).  
- **Preprocessor embutido** → tokenização automática no pipeline.  
- **`backbone.trainable = False`** → congela camadas internas, treinando apenas a cabeça (camadas densas finais).

Essa abordagem de *linear probing* é ideal para prototipagem, pois reduz drasticamente o tempo de ajuste e o uso de memória.

---

## 6. Avaliação e Métricas

A avaliação do modelo é feita após o treinamento com métricas de classificação binária:

```python
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
```

- **AUC (Área sob a Curva ROC)**: mede separabilidade entre classes.  
- **Accuracy**: proporção de predições corretas.  
- **Matriz de confusão**: quantifica acertos e erros por classe.

Em cenários com treinamento leve (`1 época`, `MAX_LEN=32`, `~3k exemplos`), o modelo atinge:

| Métrica | Valor aproximado |
|----------|------------------|
| AUC      | 0.75 – 0.82      |
| Accuracy | 0.70 – 0.80      |

Resultados bastante competitivos, considerando o baixo custo de treinamento.

---

## 7. Salvamento e Reuso do Modelo

Três formatos de salvamento foram configurados:

1. **Formato completo (`.keras`)**
   ```python
   bert_clf.save("artifacts_bot_detection/bert_final.keras")
   ```
   > Ideal para reabrir no Colab ou localmente sem reconstruir o modelo.

2. **Somente pesos (`.h5`)**
   ```python
   bert_clf.save_weights("artifacts_bot_detection/bert_weights.h5")
   ```
   > Mais leve, mas requer recriar a arquitetura antes de carregar os pesos.

3. **Exportação para produção (TensorFlow SavedModel)**
   ```python
   bert_clf.export("artifacts_bot_detection/bert_savedmodel")
   ```
   > Compatível com TensorFlow Serving e APIs (FastAPI, Flask etc).

---

## 8. Considerações

O notebook `Twitter_Bot_Detection_BERT_Colab.ipynb` comprova que é possível treinar modelos **transformer de última geração (BERT)** em **ambientes de hardware limitado**, mantendo boa precisão e desempenho.  

As principais conclusões são:

- **Eficiência:** com subamostragem e backbone congelado, o modelo treina em poucos minutos.  
- **Desempenho sólido:** o BERT, mesmo parcialmente treinado, consegue identificar padrões textuais típicos de bots.  
- **Facilidade de uso:** o KerasNLP simplifica o pipeline de tokenização e treinamento.  
- **Reprodutibilidade:** o modelo final é salvo e pode ser facilmente recarregado.  

### Recomendações futuras:
1. **Fine-tuning completo** do BERT (descongelando camadas superiores).  
2. Aumento progressivo do `MAX_LEN` e do tamanho do dataset.  
3. Teste com variantes multilíngues (`bert_multilingual_base`).  
4. Implementar **interpretação de predições** (attention weights, LIME, SHAP).  
5. Realizar **validação cruzada (k-fold)** para maior robustez estatística.

---

## 9. Resumo Final

| Aspecto | Descrição |
|----------|------------|
| **Modelo** | BERT (`bert_base_en`, KerasNLP) |
| **Tarefa** | Classificação binária (bot vs humano) |
| **Técnica** | Linear Probing (backbone congelado) |
| **Ambiente** | Google Colab (GPU) |
| **Otimizações** | XLA, Mixed Precision, Subamostragem, MAX_LEN reduzido |
| **Tempo total de execução** | ~6 a 10 minutos |
| **Resultados** | AUC ~0.8, Accuracy ~0.75 |
| **Formatos salvos** | `.keras`, `.h5`, `SavedModel` |

---

### 📘 Conclusão Final

O pipeline implementado é **eficiente, reproduzível e escalável**.  
Mesmo em condições de hardware restrito, o uso inteligente do BERT via **KerasNLP** mostrou-se uma solução poderosa para tarefas de **detecção de bots** em mídias sociais.  
Esse projeto serve como base sólida para evoluir para modelos de **linguagem contextual mais profundos**, **datasets maiores** e **aplicações reais de detecção automática** de comportamentos suspeitos em redes sociais.
