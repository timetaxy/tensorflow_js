# 딥러닝
https://codingapple.com/course/python-deep-learning/

미지수
weight, bias편향

딥러닝은 히든레이어 퍼셉트론 모델
perceptron, hiddenlayer, 뉴럴네트워크
feature extraction 특성추출
편균제곱오차, 분산?, loss function 
예측값표기 ^

식 전개식 히든레이어 크게 차이 없음
활성함수, 변형 sigmoid, hyperbolic tangent, rectifiedLinearUnits
비선형적인 예측가능
활성함수가 없으면 히든레이어가 무용지물

경사하강법, 접선의 기울기 만큼 조정
부분구간최소값과 구분이 안되는 경우 해결책 : 러닝레이트 더 많이 움직임 0.00001 정도부터 시작
러닝레이트 옵티마이저:momentum, adagrad, rmsdrop, adadelta, adam 일반적으로 adam 문안

import tensorflow as tf
aa = tf.constant([1,2,3])
tf.add()
substract
divide
multiply
matmul 행렬곱
tf.zeros(10)

print(aa.shape)
세잎: [1,2] 원소 2개인 배열이 1개, 뒤에서부터 해석

w = tf.Variable(1.0)
w.numpy()
w.assign(2)
변경할 값

경사하강법
def 손실함수(): return tf.square(실제값-예측값)
def 손실함수(): 
	예측값=키 * a * b
	return tf.square(260-예측값)
a=tf.Variable(0.1)
b=tf.Variable(0.2)
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
for i in range(300):
	opt.minimize(손실함수, var_list=[a,b])
	print(a.numpy(),b.numpy())

keras 텐서플로우 안의 유틸
1.딥러닝모델디자인하기
model=tf.keras.models.Sequential([
tf.keras.layer.Dense(64, activation='tanh'),
tf.keras.layer.Dense(128, activation='relu'),
tf.keras.layer.Dense(1, activation='sigmoid'),
])
//결과는 히든레이어1 결과판정위해, 불린결과 sigmoid

2.모델컴파일
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# 전처리

data = pd.read_csv('aaa.csv')
빈데이터 평균값 넣거나 삭제
print(data.isnull().sum())
data=data.dropna()
print(data.isnull().sum())
or
data.fillna(평균값)
exit()//일단 프린트결과확인 위해
데이터=data['admit'].values

x = []
for i,rows in data.iterrows():
	x.append([rows['a'],rows['b'],rows['c']])


3.모델학습(fit)시키기
model.fit(np.array(x), np.array(y), epochs=10)

에측 = model.predict([[data1],[data2]])
print(예측)
