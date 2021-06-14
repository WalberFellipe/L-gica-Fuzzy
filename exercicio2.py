
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



sell = ctrl.Antecedent(np.arange(0, 1.01, .01), 'sell')
price = ctrl.Antecedent(np.arange(100000, 1001000, 1000), 'price')

quality = ctrl.Consequent(np.arange(0, 100.1, .1), 'quality')

#VENDA
sell['low'] = fuzz.trimf(sell.universe, [0, 0.25, 0.5])
sell['moderate'] = fuzz.trimf(sell.universe, [0.3,0.5,0.7])
sell['high'] = fuzz.trimf(sell.universe, [0.6,0.8,1])

#PREÇO
price['lower'] = fuzz.trimf(price.universe, [100000, 200000,300000])
price['normal'] = fuzz.trimf(price.universe, [250000, 500000, 700000])
price['high'] = fuzz.trimf(price.universe, [600000, 800000, 1000000])

#QUALIDADE
quality['worst'] = fuzz.trapmf(quality.universe, [0, 25, 45,60])
quality['average'] = fuzz.trapmf(quality.universe, [40, 50, 70,80])
quality['good'] = fuzz.trapmf(quality.universe, [75, 80,95,100])

# VISUALIZAÇÃO DAS FUNÇÕES DE PERTINÊNCIA
#VENDA
sell.view()

#PREÇO
price.view()

#QUALIDADE
quality.view()

# regra 1 - se probabilidade de venda é baixa, então a qualidade é ruim
regra1 = ctrl.Rule(sell['low'], quality['worst'])

# regra 2 - se probabilidade de venda é média ou o preço é médio, então qualidade é mediana
regra2 = ctrl.Rule(sell['moderate'] | price['normal'], quality['average'])

# regra 3 - se probabilidade de venda é alta e o preço é alto, então qualidade é boa
regra3 = ctrl.Rule(sell['high'] & price['high'], quality['good'])

# regra 4 - se probabilidade sell é médio ou o preço é baixo, então quality é mediana
regra4 = ctrl.Rule(sell['moderate'] | price['lower'], quality['average'])

# regra 5 - se probabilidade sell é low e o preço é high, então quality é mediana
regra5 = ctrl.Rule(sell['low'] & price['high'], quality['average'])

imovel_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])
engine = ctrl.ControlSystemSimulation(imovel_ctrl)

# passa as predições dos modelos para suas respectivas variáveis de entrada
engine.input['sell'] = 0.8
engine.input['price'] = 150000

# calcula a saída do sistema de controle fuzzy
engine.compute()

# retorna o valor crisp e o gráfico mostrando-o
print(engine.output['quality'])
quality.view(sim=engine)