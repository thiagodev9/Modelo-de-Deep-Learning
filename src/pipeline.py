# Imports

# Imports para manipulação e visualização de dados
import numpy as np
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt
import seaborn as sns

# Imports para pré-processamento e avaliação
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Imports para Deep Learning
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Nadam



tf.get_logger().setLevel('ERROR')

# Carregar o dataset
dataset = pd.read_csv('dataset.csv')

# Mostrar dimensões do dataset
print("Dimensões do dataset:", dataset.shape)
print("\nPrimeiras 5 linhas do dataset:")
print(dataset.head())

# Visualizar as primeiras linhas do dataset

# Análise exploratória
# Números da coluna
print(dataset.columns)
print(dataset.dtypes)

# Colunas tipo string
for column, series in dataset.items():
    if str(type(series[0])) == "<class 'str'>":
        print(f"Coluna tipo string é {column}:")


# Checar se há valores nulos
print(f"Valores nulos: {dataset.isnull().sum()}")

# Checar se há valores duplicados
print(f"Valores duplicados: {dataset.duplicated().sum()}")


# Limpeza e transformação dos dados
# Remover colunas company
dataset = dataset.drop(columns=['company'])

# Remover coluna country
dataset = dataset.drop(columns=['country'])

# Remover valores nulos
dataset = dataset.dropna()

# Verificar o shape do dataset
print(f"Shape do dataset: {dataset.shape}")

# Verificar se há valores ausentes
print(f"Valores ausentes: {dataset.isnull().sum()}")

# Endiconding da variável com mês de chegada
# Valores únicos para a coluna de mês da chegada
print(dataset['arrival_date_month'].unique())

# Criar uma nova coluna com os valores codificados
dataset['arrival_month_numerical'] = dataset['arrival_date_month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
})

# Total de registros para cada mês
print(dataset['arrival_month_numerical'].value_counts())

# Não preciso mais da coluna que tem mês em texto
dataset = dataset.drop(columns=['arrival_date_month'])

# Endiconding da variável com tipo de quarto reservado e tipo de quarto ocupado
# Valores únicos para a coluna de tipo de quarto reservado
print(dataset['reserved_room_type'].unique())

# Valores únicos para a coluna de tipo de quarto ocupado
print(dataset['assigned_room_type'].unique())

# Vamos criar um dicionário para mapear os valores reservados e ocupados
reserved_room_type_mapping = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'M': 11, 'P': 12, 'S': 13, 'V': 14
}

# Aplicar o mapeamento
dataset['reserved_room_type_numerical'] = dataset['reserved_room_type'].map(reserved_room_type_mapping)
dataset['assigned_room_type_numerical'] = dataset['assigned_room_type'].map(reserved_room_type_mapping)

# Remover as colunas que tem representação de texto
dataset = dataset.drop(columns=['reserved_room_type', 'assigned_room_type'])

# Endiconding da variável com tipo de depósito
# Valores únicos para a coluna de tipo de depósito
print(dataset['deposit_type'].unique())

# Vamos criar um dicionário para mapear os valores
deposit_type_mapping = {
    'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2
}

# Aplicar o mapeamento
dataset['deposit_type_numerical'] = dataset['deposit_type'].map(deposit_type_mapping)

# Remover a coluna que tem representação de texto
dataset = dataset.drop(columns=['deposit_type'])

# Endiconding da variável com tipo de cliente
# Valores únicos para a coluna de tipo de cliente
print(dataset['customer_type'].unique())

# Vamos criar um dicionário para mapear os valores
customer_type_mapping = {
    'Transient': 0, 'Contract': 1, 'Transient-Party': 2
}   

# Aplicar o mapeamento
dataset['customer_type_numerical'] = dataset['customer_type'].map(customer_type_mapping)

# Remover a coluna que tem representação de texto
dataset = dataset.drop(columns=['customer_type'])

# Endiconding da variável com tipo de hotel     
# Valores únicos para a coluna de tipo de hotel
print(dataset['hotel'].unique())

# Vamos criar um dicionário para mapear os valores
hotel_mapping = {
    'City Hotel': 0, 'Resort Hotel': 1
}

# Aplicar o mapeamento
dataset['hotel_numerical'] = dataset['hotel'].map(hotel_mapping)

# Remover a coluna que tem representação de texto
dataset = dataset.drop(columns=['hotel'])

# Endiconding da variável com tipo de refeicao  
# Valores únicos para a coluna de tipo de refeicao
print(dataset['meal'].unique())

# Vamos criar um dicionário para mapear os valores
meal_mapping = {
    'SC': 0, 'HB': 1, 'FB': 2, 'Undefined': -1
}

# Aplicar o mapeamento
dataset['meal_numerical'] = dataset['meal'].map(meal_mapping)

# Remover a coluna que tem representação de texto
dataset = dataset.drop(columns=['meal'])

# Endiconding da variável com tipo de segmento de mercado
# Valores únicos para a coluna de tipo de segmento de mercado
print(dataset['market_segment'].unique())
print(dataset['distribution_channel'].unique())
# Vamos criar um dicionário para mapear os valores
market_segment_mapping = {
    'Direct': 0, 'Corporate': 1, 'GDS': 2, 'Undefined': -1
}   

# Aplicar o mapeamento
dataset['market_segment_numerical'] = dataset['market_segment'].map(market_segment_mapping)
dataset['distribution_channel_numerical'] = dataset['distribution_channel'].map(market_segment_mapping)

# Remover a coluna que tem representação de texto
dataset = dataset.drop(columns=['market_segment', 'distribution_channel'])

# Visualizar o dataset
print(dataset.head())

# Visualiza os tipos de dados
print(dataset.dtypes)

# Valores únicos esta será nossa variável alvo, que desejamos prever
print(dataset['is_canceled'].unique())

# Vamos criar um dicionário para mapear os valores
is_canceled_mapping = {
    0: 0, 1: 1
}

# Valores únicos
print(dataset['reservation_status'].unique())

# Vamos criar um dicionário para mapear os valores
reservation_status_mapping = {
    'Canceled': 0, 'Check-Out': 1, 'No-Show': 2
}

# Drop reservation_status
dataset = dataset.drop(columns=['reservation_status'])

# Visualizar o dataset
print(dataset.head())   

# Visualizar os tipos de dados representa data reservation_status_date
print(dataset['reservation_status_date'].dtypes)

# Convertendo para datetime
dataset['reservation_status_date'] = pd.to_datetime(dataset['reservation_status_date'])

# Drop reservation_status_date
dataset = dataset.drop(columns=['reservation_status_date'])

# Visualizar shape do dataset
print(dataset.shape)

# Visualizar os tipos de dados
print(dataset.dtypes)

# Pipeline de pré-processamento de dados
# Primeiro separamos os dados de entrada em x e y
x = dataset.drop(columns=['is_canceled'])
y = dataset['is_canceled']

# Convertemos para o tipo categórico
y = to_categorical(y, num_classes=None)

# Separamos os dados em conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=420)

# Shape dos dados
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Visualizar os tipos de dados
print(x_train.dtypes)

# Como as variáveis estão em escala diferente, vamos padronizar e deixar tudo em uma mesma escala. Fazemos isso somente com x
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Visualizar os dados
print(f'Matrix da variável de entrada:\t{x_train.shape}\nMatrix da variável de saída:\t{y_train.shape}')

# Pipeline de construção do modelo
# Vamos criar um modelo de Deep Learning com 5 camadas de dropout para evitar o overfitting e ativação softmax para a classificação

# Criando o modelo
model = Sequential()
model.add(Dense(200, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))   

# Pipeline de otimização e compilação do modelo
# Otimização com Nadam
otimizador = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# Compilando o modelo
model.compile(optimizer=otimizador, loss='categorical_crossentropy', metrics=['accuracy'])

# Criar 2 callbacks
# EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
# ReduceLROnPlateau
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)]

# Pipeline de treinamento do modelo
# Definindo o número de épocas e o batch size
epochs = 50
batch_size = 32

# Treinando o modelo
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=callbacks)  

# Visualizar a acurácia do modelo
print("\n Treinamento inicializado com sucesso! ")
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=callbacks)  

# Visualizar a acurácia do modelo
print("\n Treinamento finalizado com sucesso! ")

# Plot da acurácia do modelo
plt.figure(figsize=(10, 8))
plt.title('Acurácia do modelo')
plt.plot(history.history['accuracy'], label='Acurácia de treinamento')
plt.xlabel('Épocas')
plt.legend()
plt.grid()
plt.show()

# Plot do erro em treino
plt.figure(figsize=(10, 8))
plt.title('Erro em treino')
plt.plot(history.history['loss'], label='Erro em treino')
plt.xlabel('Épocas')
plt.legend()
plt.grid()
plt.show()

# Fazemos as previsões em previssões de classe
y_pred = (model.predict(x_test) > 0.5)

# Calculamos a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')


# Se estiver tudo certo, salvamos o modelo
import joblib
joblib.dump(scaler, 'scaler.joblib')

# Salvamos o modelo
model.save('modelo.h5')

# Deploy e pipeline de inferência com o modelo treinado
# vamos considerar que os dados atendem aos requisitos de entrada
# Novos dados de um novo cliente
# Não temos a variável alvo, isso é uma previsão

novos_dados = {
    'lead_time': [15.00],
    'arrival_date_year': [2016.00],
    'arrival_date_week_number': [23.00],
    'arrival_date_day_of_month': [30.00],
    'stays_in_weekend_nights': [2.00],
    'stays_in_week_nights': [5.00],
    'adults': [2.00],
    'children': [0.00],
    'babies': [0.00],
    'is_repeated_guest': [0.00],
    'previous_cancellations': [0.00],
    'previous_bookings_not_canceled': [0.00],
    'booking_changes': [0.00],
    'agent': [14.00],
    'days_in_waiting_list': [0.00],
    'adr': [115.84],
    'required_car_parking_spaces': [0.00],
    'total_of_special_requests': [0.00],
    'arrival_date_month_numerical': [5.00],
    'reserved_room_type_numerical': [4.00],
    'assigned_room_type_numerical': [4.00],
    'deposit_type_numerical': [0.00],
    'customer_type_numerical': [0.00],
    'hotel_numerical': [1.00],
    'meal_numerical': [0.00],
    'market_segment_numerical': [0.00],
    'distribution_channel_numerical': [0.00]
}

# Criando o DataFrame
novo_cliente = pd.DataFrame(novos_dados)

# Carregando o scaler do disco
scaler = joblib.load('scaler.joblib')

# Padronizando os dados
novo_cliente = scaler.transform(novo_cliente)

# Carregando o modelo do disco  
modelo = keras.models.load_model('modelo.h5')

# Fazendo a previsão
previsao = modelo.predict(novo_cliente)

# Exibindo a previsão
print(previsao)

# Carrega o modelo do disco
modelo_final = keras.models.load_model('modelo.h5')

# Fazendo a previsão
previsao = modelo_final.predict(novo_cliente)

# Exibindo a previsão
print(previsao) 

# Podemos entregar a previsão de classe (para modelos de classificação)
print("Esta é a previsão de cliente não cancelar a reserva:", previsao[0,0]*100)
print("Esta é a previsão de cliente cancelar a reserva:", previsao[0,1]*100)

# Verificanddo o primeiro valor do array
if previsao[0] > 0.5:
    print('O cliente vai cancelar a reserva')
else:
    print('O cliente não vai cancelar a reserva')






































