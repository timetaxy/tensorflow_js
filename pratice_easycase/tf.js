`use strict`;
const tf = require('@tensorflow/tfjs-node');
const util = require('util');
const data = { factor: [2, 2, 2, 2, 2], result: [10, 10, 10, 10, 10], future: [2] };
// const data = { factor: [2, 3, 4, 5, 6, 7], result: [400, 600, 800, 1000, 1200, 1400], future: [1] };
const dataFactor = tf.tensor(data.factor);
const dataResult = tf.tensor(data.result);

const x = tf.input({ shape: [1] });
const h1 = tf.layers.dense({ units: 1, activation: 'relu' }).apply(x);
// const h2 = tf.layers.dense({ units: 1, activation: 'relu' }).apply(x);
// const h2 = tf.layers.dense({ units: 1, activation: 'relu' }).apply(h1);
// const y = tf.layers.dense({ units: 1 }).apply(h2);
const y = tf.layers.dense({ units: 1 }).apply(h1);
const model = tf.model({ inputs: x, outputs: y });
const compileParam = { optimizer: tf.train.adamax(), loss: tf.losses.meanSquaredError };
// const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError };
model.compile(compileParam);

// const simpleFitParam = { epochs: 1 };
const fitParam = {
    epochs: 5000,
    callbacks: {
        onEpochEnd:
            function (epoch, logs) {
                console.log(`epoch:${epoch}, logs:${util.inspect(logs)}, rmse:${Math.sqrt(logs.loss)}`);
            }
    }
};
model.fit(dataFactor, dataResult, fitParam).then(function (result) {
    // model.fit(dataFactor, dataResult, simpleFitParam).then(function (result) {
    let prediction = model.predict(tf.tensor(data.future));
    // let prediction = model.predict(dataFactor);
    let tensorDetail = prediction.arraySync();
    let wb = model.getWeights();
    //[0] weight, [1] bias

    //값 가져오기
    // https://www.tensorflow.org/js/guide/tensors_operations?hl=ko
    let tempVar = util.inspect(wb[0].arraySync());
    console.log(tempVar);
    let firstW = wb[0].arraySync()[0][0];
    let firstB = wb[1].arraySync()[0];
    let firstPredictRes = data.future[0] * firstW + firstB;
    console.log(`firstWeight ${util.inspect(wb[0].arraySync())}`);
    console.log(`firstBias ${util.inspect(wb[1].arraySync())}`);
    console.log(`firstPredictRes ${firstPredictRes}`);

    console.log(`prediction detail: ${util.inspect(tensorDetail)}`);
    prediction.print();
    // console.log(util.inspect(prediction));



});

