import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

cs = ctrl.Antecedent(np.arange(4,8), 'CS')
ls = ctrl.Antecedent(np.arange(2,5), 'LS')
cp = ctrl.Antecedent(np.arange(1,7), 'CP')
lp = ctrl.Antecedent(np.arange(0,3), 'LP')

classe = ctrl.Consequent(np.arange(0, 1), 'classe')

cs['low'] = fuzz.trimf(cs.universe, [0, 0.25, 0.5])
cs['normal'] = fuzz.trimf(cs.universe, [0.3,0.5,0.7])
cs['high'] = fuzz.trimf(cs.universe, [0.6,0.8,1])

ls['low'] = fuzz.trimf(ls.universe, [0, 0.25, 0.5])
ls['normal'] = fuzz.trimf(ls.universe, [0.3,0.5,0.7])
ls['high'] = fuzz.trimf(ls.universe, [0.6,0.8,1])

cp['low'] = fuzz.trimf(cp.universe, [0, 0.25, 0.5])
cp['normal'] = fuzz.trimf(cp.universe, [0.3,0.5,0.7])
cp['high'] = fuzz.trimf(cp.universe, [0.6,0.8,1])

lp['low'] = fuzz.trimf(lp.universe, [0, 0.25, 0.5])
lp['normal'] = fuzz.trimf(lp.universe, [0.3,0.5,0.7])
lp['high'] = fuzz.trimf(lp.universe, [0.6,0.8,1])

classe['íris-versicolor'] = fuzz.trapmf(classe.universe, [0, 0.25, 0.5])
classe['íris-setosa'] = fuzz.trapmf(classe.universe, [0.3,0.5,0.7])
classe['íris-virginica'] = fuzz.trapmf(classe.universe, [0.6,0.8,1])

cs.view()
ls.view()
cp.view()
lp.view()

classe.view()

# regras
regra1 = ctrl.Rule(cs['regular'], ls['big'],cp['regular'], lp['regular'], classe['íris-versicolor'])
regra2 = ctrl.Rule(cs['small'], ls['regular'],cp['small'], lp['small'], classe['íris-setosa'])
regra3 = ctrl.Rule(cs['big'], ls['regular'],cp['big'], lp['regular'], classe['íris-virginica'])

iris_ctrl = ctrl.ControlSystem([regra1, regra2, regra3])
engine1 = ctrl.ControlSystemSimulation(imovel_ctrl)
engine2 = ctrl.ControlSystemSimulation(imovel_ctrl)

# modelo 1
engine1.input['CS'] = 5.8
engine1.input['LS'] = 4.1
engine1.input['CP'] = 3.5
engine1.input['LP'] = 1.1

# modelo 2
engine2.input['CS'] = 7.8
engine2.input['LS'] = 4
engine2.input['CP'] = 6.7
engine2.input['LP'] = 2

engine1.compute()
engine2.compute()

print(engine1.output['classe'])
print(engine2.output['classe'])
classe.view(sim=engine1)
classe.view(sim=engine2)