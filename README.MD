# Modelo de Previsão de Cancelamento de Reservas de Hotel

## Descrição
Este projeto implementa um modelo de aprendizado profundo (Deep Learning) para prever o cancelamento de reservas de hotel. O modelo utiliza dados históricos de reservas para identificar padrões e fazer previsões sobre a probabilidade de um cliente cancelar sua reserva.

## Estrutura do Projeto
O pipeline está organizado nas seguintes etapas:

### 1. Preparação dos Dados
- Carregamento do dataset
- Análise exploratória inicial
- Tratamento de valores nulos e duplicados
- Remoção de colunas desnecessárias

### 2. Pré-processamento
- Codificação (encoding) de variáveis categóricas:
  - Mês de chegada
  - Tipo de quarto
  - Tipo de depósito
  - Tipo de cliente
  - Tipo de hotel
  - Tipo de refeição
  - Segmento de mercado
- Normalização dos dados usando MinMaxScaler

### 3. Arquitetura do Modelo
O modelo de Deep Learning possui:
- 5 camadas densas
- Camadas de Dropout para evitar overfitting
- Ativação ReLU nas camadas intermediárias
- Ativação Softmax na camada de saída
- Otimizador Nadam
- Loss function: categorical_crossentropy

### 4. Treinamento
- Divisão dos dados em treino e teste (80/20)
- Implementação de callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
- 50 épocas de treinamento
- Batch size de 32

### 5. Avaliação
- Visualização da acurácia do modelo
- Plotagem do erro em treino
- Cálculo da acurácia final

### 6. Deploy
- Salvamento do modelo treinado
- Pipeline de inferência para novos dados
- Exemplo de previsão com novo cliente

## Requisitos
```python
numpy
pandas
matplotlib
seaborn
sklearn
tensorflow
keras
joblib
```

## Como Usar
1. Certifique-se de ter todos os requisitos instalados
2. Execute o script principal
3. Para fazer previsões com novos dados:
   - Prepare os dados no formato correto
   - Utilize o pipeline de inferência fornecido

## Exemplo de Uso
```python
novos_dados = {
    'lead_time': [15.00],
    'arrival_date_year': [2016.00],
    # ... outros campos necessários ...
}

novo_cliente = pd.DataFrame(novos_dados)
previsao = modelo.predict(novo_cliente)
```

## Resultados
O modelo fornece:
- Probabilidade de não cancelamento
- Probabilidade de cancelamento
- Previsão final (cancelamento ou não)

## Notas
- O modelo foi treinado com dados históricos de reservas de hotel
- A acurácia e performance podem variar dependendo dos dados de entrada
- Recomenda-se retreinamento periódico com dados mais recentes

## Autor
[Thiago]


# Modelo-de-Deep-Learning
