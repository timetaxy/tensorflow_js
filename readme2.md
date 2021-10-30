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

var X = tf.input({shape:[1]})
var Y = tf.layer.dense({units:}).apply(X)
var model = tf.model({inputs:X,outputs:Y})
var compilePParam = {optimizer:tf.train.adam(), loss:tf.losses.meanSquaredError}
model.compile(compileParam)

var fitParam = {epochs:100}
<!-- epoch 학습횟수 -->
model.fit(factor, result,fitParam).then(function(result){
    var nextTempa = [2,3,4]
    var nextFactor = tf.tensor2d(nextTempa,[nextTempa.length,1])
    var nextResult = model.predict(nextFactor)
    nextResult.print()
})