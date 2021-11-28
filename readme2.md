# js텐서플로우 < 회귀, 지도학습
 https://www.youtube.com/watch?v=5q2E3JMXTKU&list=PLuHgQVnccGMBEbPiaGs2kfQFpMmQchM-1
분류classification 회귀regression
뉴럴네트워크=인공신경망=딥러닝

라이브러리 : tensorflow, pytorch, caffe2, convnetjs, theano

지도학습 과거의 데이터에서 원인과 결과 변수 정의
1데이터준비 2모델생성 3모델학습 4모델이용

1.기존모델 실행 / 2. 기존모델 다시학습 / 3. 자신만의 모델 개발
tensorflow.org/js

https://github.com/tensorflow/tfjs-models/tree/master/mobilenet

--- basic sequence of model

var tempa = [1,2,3]
var selling =[10,20,30]
var factor = tf.tensor(degree)
var result = tf.tensor(selling)

# 원인 input shape
# 결과 y layer dense units
var X = tf.input({shape:[1]})
var Y = tf.layer.dense({units:1}).apply(X)
var model = tf.model({inputs:X,outputs:Y})
var compilePParam = {optimizer:tf.train.adam(), loss:tf.losses.meanSquaredError}
# loss function, MSE : 분산 구하기와 유사
model.compile(compileParam)

var fitParam = {epochs:100}
# epoch 학습횟수
model.fit(factor, result,fitParam).then(function(result){
    var nextTempa = [2,3,4]
    var nextFactor = tf.tensor2d(nextTempa,[nextTempa.length,1])
    var nextResult = model.predict(nextFactor)
    nextResult.print()
})

---
https://www.tensorflow.org/js/tutorials/setup

npm install @tensorflow/tfjs

https://github.com/egoing/tensorflow.js-1

// epoch 더이상 줄어들지 않을 때까지 학습

y=a*x+b
a가중치, b편향
weight, bias

model.predict(tf.tensor([20])).arraySync()[0][0]
weights=model.getWeights()
//tensor를 일반 값으로 .arraySync()
weight = weights[0].arraySync()[0][0]
bias = weights[1].arraySync()[0]
//weight*20+bias 현재 모델에서는
//모델 만들기는 많은 연산, 사용은 사칙연산 1회

-- 모델 속성
pratice5_2

--- 모델 저장
https://www.tensorflow.org/js/guide/save_load

mode.save('downloads://model1');
model.save('localstorage://my-model'); //브라우저 스토리지 저장됨

--- 레몬에이드 예제
pratice5_3

--- 로드 및 모델 사용
pratice8

---10_3
chrome 콘솔에서 실시간 실행시
fit 부분만 콘솔에서 실행 후 rmse 최소화

--10_4 복수 결과
# 모델보기
var weights=model.getWeights();
//배열 0은 w, 1은 b
weights[0].arraySync()
weights[1].arraySync()
// 결과가 2개 입력되었다면, w 는 각 원소 2개씩 배열이 생성되어 있음

-- 학습 시각화
pratice11
https://github.com/tensorflow/tfjs-vis
https://js.tensorflow.org/api_vis/1.5.1/

인자 요약보기
tfvis.show.modelSummary({name:'nm',tab:'model},model);

실시간 학습곡선
tfvis.show.history({name:'loss',tab:'history'},[{loss:10},{loss:5},{loss:1}],['loss'])

       var _history = [];
        var fitParam = { 
          epochs: 100, 
          callbacks:{
            onEpochEnd:
              function(epoch, logs){
                console.log('epoch', epoch, logs, 'RMSE=>', Math.sqrt(logs.loss));
                _history.push(logs);
                tfvis.show.history({name:'loss', tab:'역사'}, _history, ['loss']);
              }
          }
        } 

---
인공신경망 ann
pratice12_2
node, layer, input layer - hidden layer - output layer


